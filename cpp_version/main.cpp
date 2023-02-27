#include <iostream>
#include <json/json.h>
#include <unordered_map>
#include <random>
#include <fstream>


int main() {
    std::ifstream trainFile("tokenized_train.json");
    Json::Reader reader;
    Json::Value trainData;
    reader.parse(trainFile, trainData);

    size_t tokenCount = 0;
    auto vocabFreqMap = std::unordered_map<std::string, size_t>();
    for(const auto& document : trainData) {
        for(auto& token : document["text"]) {
            tokenCount++;
            if (vocabFreqMap.contains(token.asString())) {
                vocabFreqMap[token.asString()]++;
            } else {
                vocabFreqMap[token.asString()] = 1;
            }
        }
    }
    auto vocabMap = std::unordered_map<std::string, size_t>();
    auto vocab = std::vector<std::string>();

    for( const auto& entry : vocabFreqMap) {
        if(static_cast<double>(entry.second) / static_cast<double>(tokenCount) > 5e-4) {
            if(!vocabMap.contains(entry.first)) {
                vocabMap[entry.first] = vocab.size();
                vocab.push_back(entry.first);
            }
        }
    }

    auto docs = std::vector<std::string>();
    auto authors = std::vector<std::string>();
    auto authorsMap = std::unordered_map<std::string, size_t>();

    auto docIds = std::vector<size_t>();
    auto authorIds = std::vector<size_t>();
    auto tokenIds = std::vector<size_t>();

    for(size_t i = 0; i < trainData.size(); i++) {
        auto& document = trainData[static_cast<Json::ArrayIndex>(i)];
        docs.push_back(document["id"].asString());

        size_t authorId = 0;
        if(authorsMap.contains(document["author"].asString())) {
            authorId = authorsMap[document["author"].asString()];
        } else {
            authorsMap[document["author"].asString()] = authors.size();
            authorId = authors.size();
            authors.push_back(document["author"].asString());
        }
        for(auto& token : document["text"]) {
            if(vocabMap.contains(token.asString())) {
                if(vocabMap.contains(token.asString())) {
                    docIds.push_back(i);
                    authorIds.push_back(authorId);
                    tokenIds.push_back(vocabMap[token.asString()]);
                }
            }
        }
    }

    size_t N = tokenIds.size();
    size_t K = 20;
    size_t V = vocab.size();
    double alpha = 0.1;
    double beta = 0.01;

    std::uniform_int_distribution<> uniform(0, K-1);

    std::vector<size_t> Z;

    std::random_device rd;
    std::mt19937 rng(rd());
    for(size_t i = 0; i < N; i++) {
        Z.push_back(uniform(rng));
    }

    std::vector<std::vector<size_t>> cdk;
    std::vector<std::vector<size_t>> ckv;
    std::vector<size_t> cd(N, 0);
    std::vector<size_t> ck(K, 0);

    for(size_t i = 0; i < K; ++i) {
        auto v1 = std::vector<size_t>(N, 0);
        cdk.push_back(v1);
        auto v2 = std::vector<size_t>(V, 0);
        ckv.push_back(v2);
    }

    for (size_t i = 0; i < N; ++i) {
        cd[i]++;
        cdk[Z[i]][i]++;
        ckv[Z[i]][tokenIds[i]]++;
        ck[Z[i]]++;
    }

    for(size_t it = 0; it < 1000; it++) {
        size_t changes = 0;
        for(size_t i = 0; i < N; i++) {
            cdk[Z[i]][i]--;
            ckv[Z[i]][tokenIds[i]]--;
            cd[i]--;
            ck[Z[i]]--;

            std::vector<double> p;
            for(size_t j = 0; j < K; ++j) {
                double num1 = alpha + static_cast<double>(cdk[j][i]);
                double denom1 = static_cast<double>(K)*alpha + static_cast<double>(cd[i]);
                double num2 = beta + static_cast<double>(ckv[j][docIds[i]]);
                double denom2 = static_cast<double>(V)*beta + static_cast<double>(ck[j]);
                double p_temp = (num1 * num2) / (denom1 * denom2);
                p.push_back(p_temp);
            }
            std::discrete_distribution<> post(p.begin(), p.end());
            size_t new_topic = post(rng);
            if(new_topic != Z[i]) {
                changes++;
            }
            Z[i] = new_topic;
            cdk[Z[i]][i]++;
            ckv[Z[i]][docIds[i]]++;
            cd[i]++;
            ck[Z[i]]++;
        }
        if(it % 20 == 0) {
            std::cout << "iteration " << it << " complete, changed " << changes << " assignments" << std::endl;
        }
    }




    return 0;
}
