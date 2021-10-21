//
//  Machine_Learning.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/10/20.
//

#include "Machine_Learning.hpp"

void StandardScaler::fit(vector<vfloat> &data) {
    int data_size = (int)data.size();
    int feature_dim = (int)data[0].size();
    for (int d = 0; d < feature_dim; ++d) {
        float mean = 0, stdev = 0;
        for (int i = 0; i < data_size; ++i) {
            mean += data[i][d];
        }
        mean /= data_size;
        for (int i = 0; i < data_size; ++i) {
            stdev += pow((data[i][d] - mean), 2);
        }
        stdev = sqrt(stdev / (data_size - 1) + 0.000001f);
        means.push_back(mean);
        stdevs.push_back(stdev);
    }
}

vector<vfloat> StandardScaler::transform(vector<vfloat> &data) {
    vector<vfloat> data_norm;
    for (int i = 0; i < data.size(); ++i) {
        vfloat norm;
        for (int d = 0; d < means.size(); ++d) {
            norm.push_back((data[i][d] - means[d]) / stdevs[d]);
        }
        data_norm.push_back(norm);
    }
    return data_norm;
}

unordered_map<int, vector<vfloat>> GaussianNB::separate_classes(vector<vfloat> &data, vint &label) {
    unordered_map<int, vector<vfloat>> separated_classes;
    for (int i = 0; i < data.size(); ++i) {
        vfloat &feature = data[i];
        int class_name = label[i];
        separated_classes[class_name].push_back(feature);
    }
    return separated_classes;
}

void GaussianNB::summarize(unordered_map<int, vector<vfloat>> &X) {
    int total_data_num = 0;
    for (auto classes : X) {
        total_data_num += classes.second.size();
    }
    for (auto classes : X) {
        int classes_name = classes.first;
        vector<vfloat> &features = classes.second;
        vfloat mean_summary;
        vfloat stdev_summary;
        int data_size = (int)features.size();
        int feature_dim = (int)features[0].size();
        for (int d = 0; d < feature_dim; ++d) {
            float mean = 0, stdev = 0;
            for (int i = 0; i < data_size; ++i) {
                mean += features[i][d];
            }
            mean /= data_size;
            for (int i = 0; i < data_size; ++i) {
                stdev += pow((features[i][d] - mean), 2);
            }
            stdev = sqrt(stdev / (data_size - 1) + 0.000001f);
            mean_summary.push_back(mean);
            stdev_summary.push_back(stdev);
        }
        summary[classes_name] = {(float)data_size / total_data_num, mean_summary, stdev_summary};
    }
}

void GaussianNB::fit(vector<vfloat> &data, vint &label) {
    auto separated_classes = this->separate_classes(data, label);
    this->summarize(separated_classes);
}

float GaussianNB::gauss_distribution_function(float x, float mean, float stdev) {
    float exponent = exp(-(pow(x - mean, 2) / (2 * pow(stdev, 2))));
    return exponent / (sqrt(2 * M_PI) * stdev);
}

int GaussianNB::predict(vfloat &data) {
    unordered_map<int, float> joint_proba;
    for (auto classes : summary) {
        int classes_name = classes.first;
        float likelihood = 1;
        auto class_summary = classes.second;
        auto mean_list = get<1>(class_summary);
        auto stdev_list = get<2>(class_summary);
        int total_feature = (int)mean_list.size();
        for (int i = 0; i < total_feature; ++i) {
            float feature = data[i];
            float mean = mean_list[i];
            float stdev = stdev_list[i];
            float normal_proba = this->gauss_distribution_function(feature, mean, stdev);
            likelihood *= normal_proba;
        }
        float prior_proba = get<0>(class_summary);
        joint_proba[classes_name] = likelihood * prior_proba;
    }
    
    auto pred = *max_element(joint_proba.begin(), joint_proba.end(), [](const pair<int, float> &p1, const pair<int, float> &p2) {return p1.second < p2.second;});
    return pred.first;
}

vint GaussianNB::predict(vector<vfloat> &data) {
    vint pred;
    for (int i = 0; i < data.size(); ++i) {
        pred.push_back(this->predict(data[i]));
    }
    return pred;
}

float accuracy_score(vint &pred, vint &truth) {
    int correct = 0, total = (int)truth.size();
    for (int i = 0; i < total; ++i) {
        if (pred[i] == truth[i])
            ++correct;
    }
    return (float)correct / total;
}
