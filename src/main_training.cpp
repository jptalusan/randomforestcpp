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
int train(Utils::Configs c);
int getClassNumberFromHistogram(int numberOfClasses, const float* histogram);
void distributedTest(Utils::Configs &c);
void centralizedTest(Utils::Configs &c);
void centralizedCrossValidate(Utils::Configs &c, int runs);
int train_for_crossval(Utils::Configs c, std::vector<RTs::Sample> &samples);
void centralizedTest_for_crossval(Utils::Configs c, std::vector<RTs::Sample> &samples);

int myrandom (int i) { return std::rand()%i;}

int main(int argc, char *argv[]){
    Utils::Json *json = new Utils::Json();
    Utils::Configs c = json->parseJsonFile("configs.json");
    // Utils::Configs c = json->parseJsonFile("iris.json");

    if (argc < 2) {
        std::cout << "rf_exe [test|train] [cent|dist]" << std::endl;
    } else {
        std::string arg1(argv[1]);
        std::string arg2 = "";
        if (argc > 2) {
            arg2 = argv[2];
        }
        Utils::Timer* t = new Utils::Timer();
        std::cout << argc << ": " << arg1 << " " << arg2 << " " << std::endl;
        if (arg1 == "train") {
            t->start();
            train(c);
            t->stop();
        } else if (arg1 == "test" && arg2 == "dist" && argc == 3) {
            t->start();
            distributedTest(c);
            t->stop();
        } else if (arg1 == "test" && arg2 == "cent" && argc == 3) {
            t->start();
            centralizedTest(c);
            t->stop();
        } else if (arg1 == "crossval") {
            t->start();
            centralizedCrossValidate(c, 3);
            t->stop();
        } else {
            std::cout << "rf_exe [test|train]" << std::endl;
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
        // std::cout << '\n';

        // for (std::vector<std::string>::iterator it=vtest.begin(); it!=vtest.end(); ++it)
        //   std::cout << ' ' << *it;
    }
}

void centralizedTest(Utils::Configs &c) {
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
    std::vector<RTs::Sample> samples = p->readCSVToSamples(c.inputFile);

    int score = 0;
    for (unsigned int i = 0; i < samples.size(); ++i) {
        RTs::Feature f = samples[i].feature_vec;
        const float* histo = rts_forest.EstimateClass(f);
        int classification = getClassNumberFromHistogram(c.numClass, histo);
        // std::cout << "classification: " << classification << std::endl;
        std::cout << "Class: " << classification << std::endl;
        if (classification == samples[i].label) {
            ++score;
        }
    }
    float f = (float) score / (float) samples.size();
    std::cout << score << "/" << samples.size() << ": " << f << std::endl;
}

void distributedTest(Utils::Configs &c) {
    // Utils::Json *json = new Utils::Json();
    // Utils::Configs c = json->parseJsonFile("configs.json");

    std::vector<std::string> nodeList = c.nodeList;

    Utils::SCP *u = new Utils::SCP();
    u->setNodeList(nodeList);
    u->getFiles();
    //u->deleteFiles();
    
    //Have to loop through all of the node list
    std::vector<std::vector<int>> scoreVectors;
    std::vector<int> correctLabel;
    std::vector<RTs::Forest> randomForests;
    //Assume to read RTs_Forest.txt
    char dir[255];
    getcwd(dir,255);
    std::cout << dir << std::endl;


    for (unsigned int i = 0; i < nodeList.size(); ++i) {
        RTs::Forest rts_forest;
        std::stringstream ss;
        ss << "RTs_Forest_" << i << ".txt";
        rts_forest.Load(ss.str());
        randomForests.push_back(rts_forest);
        ss.str(std::string());
    }
    //todo: process the rts_forest, load fxn already created the node
    // read the csv file here
    std::vector<RTs::Sample> samples = getSamples(c);
    

    //Too many loops for testing
    //Need to change checkScores func to just accept the samples vector (too large? const)
    std::for_each(samples.begin(), samples.end(), [&](const RTs::Sample& s) {
        correctLabel.push_back(s.label);
    });

    for (unsigned int i = 0; i < samples.size(); ++i) {
        std::vector<int> nodeListSamples(0, 3);
        scoreVectors.push_back(nodeListSamples);
        for (unsigned int j = 0; j < nodeList.size(); ++j) {
            RTs::Feature f = samples[i].feature_vec;
            const float* histo = randomForests[j].EstimateClass(f);

            //std::cout << getClassNumberFromHistogram(10, histo) << std::endl;

            scoreVectors[i].push_back(getClassNumberFromHistogram(c.numClass, histo));
        }
    }

    Utils::TallyScores *ts = new Utils::TallyScores();
    ts->checkScores(correctLabel, scoreVectors);
    //u->deleteLocalFiles();
    delete u;
    delete ts;
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
int train(Utils::Configs c) {
    char dir[255];
    getcwd(dir,255);
    std::cout << dir << std::endl;

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
    //

    // std::cout << "1_Randomized Forest generation" << std::endl;
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

int train_for_crossval(Utils::Configs c, std::vector<RTs::Sample> &samples) {
    char dir[255];
    getcwd(dir,255);
    std::cout << dir << std::endl;

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