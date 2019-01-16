#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <float.h>
#include <algorithm>
#include "utils.hpp"
#include "rts_forest.hpp"
#include "rts_tree.hpp"

#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include <chrono>
#include <thread>

//#define DEBUG
int train(Utils::Configs c, std::string label);
void centralizedTest(Utils::Configs &c, std::string label);

void centralizedCrossValidate(Utils::Configs &c, int runs);
int train_for_crossval(Utils::Configs c, std::vector<RTs::Sample> &samples);
void centralizedTest_for_crossval(Utils::Configs c, std::vector<RTs::Sample> &samples);

int getClassNumberFromHistogram(int numberOfClasses, const float* histogram);
int myrandom (int i) { return std::rand()%i;}

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

// Add method to edit json config file, ie. number of trees and % of data
int main(int argc, char *argv[]){
    Utils::Json *json = new Utils::Json();
    if (argc < 2) {
        std::cout << "rf_exe [train|classify] [all|ACTIVITY_NAME]" << std::endl;
        std::cout << "-t [number of trees per forest]" << std::endl;
        std::cout << "-i [input csv file for classification]" << std::endl;
        std::cout << "-c [config file]" << std::endl;
    } else {
        std::string arg1(argv[1]);
        std::string arg2 = "";
        int numOfTrees = 3;
        std::string csvFile = "SmartHome/" + arg2 + ".csv";
        std::string config = "SmartHome.json";

        if (argc > 2) {
            arg2 = argv[2];
        }

        InputParser input(argc, argv);
        if(input.cmdOptionExists("-t")){
            // Do stuff
            std::string::size_type sz;
            const std::string &filename = input.getCmdOption("-t");
            numOfTrees = std::stoi(filename, &sz);
            std::cout << "Number of trees: " << numOfTrees << std::endl;
        }

        if(input.cmdOptionExists("-i")){
            // Do stuff
            csvFile = input.getCmdOption("-i");
            std::cout << "Input CSV File: " << csvFile << std::endl;
        }

        if(input.cmdOptionExists("-c")){
            // Do stuff
            config = input.getCmdOption("-c");
            std::cout << "Config file: " << config << std::endl;
        }

        std::cout << argc << ": " << arg1 << " " << arg2 << std::endl;
        Utils::Timer* t = new Utils::Timer();

        Utils::Configs c = json->parseJsonFile(config);
        c.setInputFile(csvFile);
        c.setNumTrees(numOfTrees);

        if (arg1 == "train" && arg2 == "all") {
            std::vector<std::string> labels = 
            { "Bathing", "ReadingBook", "UsingSmartphone", "WorkingOnPC",
            "CleaningRoom", "CleaningBathroom", "PlayingGame", "WashingDishes", 
            "Eating", "Sleeping", "StayingAtHome", "Dressing", 
            "WatchingTelevision", "Laundry", "PersonalHygiene", "UsingToilet", 
            "Cooking" };

            t->start();

            for (auto label : labels) {
                train(c, label);
            }

            t->stop();
        } else if (arg1 == "train" && arg2 != "all") {
            std::string label = arg2;

            t->start();

            train(c, label);

            t->stop();
        } else if (arg1 == "classify" && arg2 == "all") {
            std::vector<std::string> labels = 
            { "Bathing", "ReadingBook", "UsingSmartphone", "WorkingOnPC",
            "CleaningRoom", "CleaningBathroom", "PlayingGame", "WashingDishes", 
            "Eating", "Sleeping", "StayingAtHome", "Dressing", 
            "WatchingTelevision", "Laundry", "PersonalHygiene", "UsingToilet", 
            "Cooking" };

            t->start();

            for (auto label : labels) {
                centralizedTest(c, label);
            }

            t->stop();
        } else if (arg1 == "classify" && arg2 != "all") {
            std::string label = arg2;

            t->start();

            centralizedTest(c, label);

            t->stop();
        } else if (arg1 == "crossval" && argc == 3) {
            std::string label = arg2;
            t->start();

            std::string filepath = "SmartHome.json";
            Utils::Configs c = json->parseJsonFile(filepath);
            c.setInputFile("SmartHome/" + label + ".csv");
            centralizedCrossValidate(c, 3);

            t->stop();
        } else {
            std::cout << "rf_exe [train|classify] [all|ACTIVITY_NAME]" << std::endl;
        }
        delete t;
    }
    return 0;
}

std::vector<RTs::Sample> getSamples(Utils::Configs &c) {
    Utils::Parser *p = new Utils::Parser();
    p->setClassColumn(c.labelColumn);
    return p->readCSVToSamples(c.inputFile);
}

void centralizedCrossValidate(Utils::Configs &c, int runs) {
/*
1. get samples
2. randomize samples
3. split into 2, train and test vectors
4. train RF using train vector
5. run test on test vectors
6. loop again
*/
    std::srand ( unsigned ( std::time(0) ) );
    Utils::Parser *p = new Utils::Parser();

    // Utils::Json *json = new Utils::Json();
    // Utils::Configs c = json->parseJsonFile("configs.json");

    p->setClassColumn(c.labelColumn);
    //p->setNumberOfFeatures(54);
    std::vector<RTs::Sample> samples = p->readCSVToSamples(c.inputFile);
    std::vector<RTs::Sample> orig_samples = samples;

    for (int i = 0; i < runs; ++i) {
        std::random_shuffle ( samples.begin(), samples.end(), myrandom );
        int totalSize = samples.size();
        int trainSize = totalSize * 0.75;

        std::vector<RTs::Sample>::const_iterator first = samples.begin();
        std::vector<RTs::Sample>::const_iterator last = samples.begin() + trainSize;
        std::vector<RTs::Sample> vtrain(first, last);
        
        // std::cout << '\n';

        // for (std::vector<std::string>::iterator it=vtrain.begin(); it!=vtrain.end(); ++it)
        //   std::cout << ' ' << *it;

        std::vector<RTs::Sample>::const_iterator firstt = samples.begin() + trainSize;
        std::vector<RTs::Sample>::const_iterator lastt = samples.end();
        std::vector<RTs::Sample> vtest(firstt, lastt);

        train_for_crossval(c, vtrain);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        centralizedTest_for_crossval(c, vtest);
    }
}

void centralizedTest(Utils::Configs &c, std::string label) {
    char dir[255];
    getcwd(dir,255);
    RTs::Forest rts_forest;
    std::ofstream outfile;

    std::stringstream ss;
    ss << "SmartHomeModels/" + label + "_forest.txt";
    rts_forest.Load(ss.str());

    Utils::Parser *p = new Utils::Parser();

    p->setClassColumn(c.labelColumn);
    //p->setNumberOfFeatures(54);
    std::vector<RTs::Sample> samples = p->readCSVToSamples(c.inputFile);

    outfile.open("output.csv", std::ios_base::app);
    outfile << label;
    std::cout << label;
    int score = 0;
    for (unsigned int i = 0; i < samples.size(); ++i) {
        RTs::Feature f = samples[i].feature_vec;
        rts_forest.EstimateClass(f);

        // rts_forest.EstimateClass(f);
        std::vector<int> classifications = rts_forest.getTreeClassifications();
        for (std::vector<int>::iterator it=classifications.begin(); it!=classifications.end(); ++it) {
            std::cout << ',' << *it;
            outfile << "," << *it;
        }
        std::cout << std::endl;
        outfile << std::endl;

        std::map<int, int> mydict = {};
        int cnt = 0;
        int classification = 0;

        for (auto&& item : classifications) {
            mydict[item] = mydict.emplace(item, 0).first->second + 1;
            if (mydict[item] >= cnt) {
                std::tie(cnt, classification) = std::tie(mydict[item], item);
            }
        }

        // std::cout << "Class: " << classification << std::endl;


        if (classification == samples[i].label) {
            ++score;
        }
    }
    // float f = (float) score / (float) samples.size();
    // std::cout << score << "/" << samples.size() << ": " << f << std::endl;
}

int getClassNumberFromHistogram(int numberOfClasses, const float* histogram) {
    float biggest = -FLT_MAX;
    int index = -1;

    //since classes start at 1
    for (int i = 0; i < numberOfClasses; ++i) {
        float f = *(histogram + i);
        if (f > biggest) {
            index = i;
            biggest = f;
        }
    }
    return index;
}

//TODO: make arguments adjustable via argv and transfer code to pi to start distribution
int train(Utils::Configs c, std::string label) {
    char dir[255];
    getcwd(dir,255);

    std::vector<RTs::Sample> samples = getSamples(c);
    //
    // Randomized Forest 生成
    //
    //  numClass = 10  //学習に用いるクラス
    //  numTrees = 5 //木の数
    //  maxDepth = 10 //木の深さ
    //  featureTrials = 50 //分岐ノード候補の数
    //  thresholdTrials = 5  //分岐ノード閾値検索の候補の数
    //  dataPerTree = .25f  //サブセットに分けるデータの割合

    RTs::Forest rts_forest;
    if(!rts_forest.Learn(
    	c.numClass, 
    	c.numTrees, 
    	c.maxDepth, 
    	c.featureTrials,
    	c.thresholdTrials, 
    	c.dataPerTree, 
    	samples)){
        printf("Randomized Forest Failed generation\n");
        std::cerr << "RTs::Forest::Learn() failed." << std::endl;
        std::cerr.flush();
        return 1;
    }

    //
    // 学習結果の保存
    //

    // std::cout << "2_Saving the learning result" << std::endl;
    std::string filepath = "SmartHomeModels/" + label + "_forest.txt";
    if(rts_forest.Save(filepath) == false){
        std::cerr << "RTs::Forest::Save() failed." << std::endl;
        std::cerr.flush();
        return 1;
    }

    return 0;
}

int train_for_crossval(Utils::Configs c, std::vector<RTs::Sample> &samples) {
    char dir[255];
    getcwd(dir,255);
    // std::cout << dir << std::endl;

    std::cout << "1_Randomized Forest generation" << std::endl;
    RTs::Forest rts_forest;
    if(!rts_forest.Learn(
        c.numClass, 
        c.numTrees, 
        c.maxDepth, 
        c.featureTrials,
        c.thresholdTrials, 
        c.dataPerTree, 
        samples)){
        printf("Randomized Forest Failed generation\n");
        std::cerr << "RTs::Forest::Learn() failed." << std::endl;
        std::cerr.flush();
        return 1;
    }

    //
    // 学習結果の保存
    //

    std::cout << "2_Saving the learning result" << std::endl;
    if(rts_forest.Save("RTs_Forest.txt") == false){
        std::cerr << "RTs::Forest::Save() failed." << std::endl;
        std::cerr.flush();
        return 1;
    }

    return 0;
}

void centralizedTest_for_crossval(Utils::Configs c, std::vector<RTs::Sample> &samples) {
    char dir[255];
    getcwd(dir,255);
    std::cout << dir << std::endl;

    RTs::Forest rts_forest;
    std::stringstream ss;
    ss << "RTs_Forest.txt";
    rts_forest.Load(ss.str());

    Utils::Parser *p = new Utils::Parser();

    // Utils::Json *json = new Utils::Json();
    // Utils::Configs c = json->parseJsonFile("configs.json");

    p->setClassColumn(c.labelColumn);
    //p->setNumberOfFeatures(54);
    // std::vector<RTs::Sample> samples = p->readCSVToSamples(c.inputFile);

    int score = 0;
    for (unsigned int i = 0; i < samples.size(); ++i) {
        RTs::Feature f = samples[i].feature_vec;
        // const float* histo = rts_forest.EstimateClass(f);
        // int classification = getClassNumberFromHistogram(c.numClass, histo);
        rts_forest.EstimateClass(f);
        std::vector<int> classifications = rts_forest.getTreeClassifications();
        for (std::vector<int>::iterator it=classifications.begin(); it!=classifications.end(); ++it)
            std::cout << ' ' << *it;
        std::cout << std::endl;

// Count most frequent
        std::map<int, int> mydict = {};
        int cnt = 0;
        int classification = 0;  // in Python you made this a string '', which seems like a bug

        for (auto&& item : classifications) {
            mydict[item] = mydict.emplace(item, 0).first->second + 1;
            if (mydict[item] >= cnt) {
                std::tie(cnt, classification) = std::tie(mydict[item], item);
            }
        }

        std::cout << "Class: " << classification << std::endl;


        // std::cout << "Class: " << classification << std::endl;
        if (classification == samples[i].label) {
            ++score;
        }
    }
    float f = (float) score / (float) samples.size();
    std::cout << score << "/" << samples.size() << ": " << f << std::endl;
}