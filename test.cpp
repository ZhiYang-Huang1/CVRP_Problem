#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <limits>
#include <map>
#include <chrono>
#include <unordered_set>
#include <mutex>

using namespace std;

// 客户节点结构
struct Node {
    int id;
    double x, y;
    int demand;
};

// 蚂蚁结构
struct Ant {
    vector<vector<int>> routes;
    double totalDistance;
    int usedVehicles;
};

// CVRP问题类
class CVRP {
private:
    string instanceName;  // 实例名称
    int dimension;        // 节点数量（包括仓库）
    int capacity;         // 车辆容量
    vector<Node> nodes;   // 所有节点（包括仓库）
    vector<vector<double>> distances;  // 距离矩阵
    int depot;            // 仓库节点ID

    // 蚁群算法参数
    int antCount;         // 蚂蚁数量
    int maxIterations;    // 最大迭代次数
    double alpha;         // 信息素重要程度
    double beta;          // 启发式因子重要程度
    double rho;           // 信息素蒸发率
    double q0;            // 状态转移规则的参数
    vector<vector<double>> pheromone;  // 信息素矩阵
    vector<vector<double>> heuristic;  // 启发式信息矩阵

    // 缓存计算结果
    vector<vector<double>> pheromone_power;  // 信息素幂矩阵
    vector<vector<double>> heuristic_power;  // 启发式信息幂矩阵

    // 随机数生成
    mt19937 rng;          // 随机数生成器
    uniform_real_distribution<double> dist;  // 均匀分布随机数

    // 局部搜索参数
    bool useLocalSearch;  // 是否启用局部搜索
    int maxLocalSearchIterations;  // 局部搜索最大迭代次数

    // 评估计数器
    int evaluationCount;
    int maxEvaluations;
    std::mutex evaluationMutex; // 添加互斥锁保护计数器
    Ant currentAnt;
public:
    Ant bestAnt;
    CVRP() : rng(chrono::high_resolution_clock::now().time_since_epoch().count()), dist(0.0, 1.0) {
        antCount = 15;                // 减少蚂蚁数量，提高效率
        maxIterations = 10000;
        alpha = 1.0;
        beta = 4;                   // 增加启发式信息权重
        rho = 0.1;
        q0 = 0.1;                     // 增加利用率
        depot = 1;
        useLocalSearch = true;        // 启用局部搜索
        maxLocalSearchIterations = 100;

        // 初始化评估计数器
        evaluationCount = 0;
        maxEvaluations = 50000;
    }
    Ant getCurrentAnt() const {
        return currentAnt;
    }
    double getCurrentSolutionCost() const {
        return currentAnt.totalDistance;
    }
    bool readInstance(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Cannot open file: " << filename << endl;
            return false;
        }

        string line;
        bool readingCoords = false;
        bool readingDemands = false;

        while (getline(file, line)) {
            if (line.find("NAME") != string::npos) {
                instanceName = line.substr(line.find(":") + 1);
                instanceName = instanceName.substr(instanceName.find_first_not_of(" \t"));
            }
            else if (line.find("DIMENSION") != string::npos) {
                string dimStr = line.substr(line.find(":") + 1);
                dimension = stoi(dimStr);
                nodes.resize(dimension + 1);
            }
            else if (line.find("CAPACITY") != string::npos) {
                string capStr = line.substr(line.find(":") + 1);
                capacity = stoi(capStr);
            }
            else if (line.find("NODE_COORD_SECTION") != string::npos) {
                readingCoords = true;
                readingDemands = false;
                continue;
            }
            else if (line.find("DEMAND_SECTION") != string::npos) {
                readingCoords = false;
                readingDemands = true;
                continue;
            }
            else if (line.find("DEPOT_SECTION") != string::npos) {
                readingCoords = false;
                readingDemands = false;
                continue;
            }
            else if (line.find("EOF") != string::npos) {
                break;
            }

            if (readingCoords) {
                istringstream iss(line);
                int id;
                double x, y;
                if (iss >> id >> x >> y) {
                    nodes[id].id = id;
                    nodes[id].x = x;
                    nodes[id].y = y;
                }
            }
            else if (readingDemands) {
                istringstream iss(line);
                int id, demand;
                if (iss >> id >> demand) {
                    nodes[id].demand = demand;
                }
            }
        }

        file.close();
        calculateDistances();
        initializeMatrices();
        return true;
    }

    void calculateDistances() {
        distances.resize(dimension + 1, vector<double>(dimension + 1, 0.0));
        for (int i = 1; i <= dimension; i++) {
            for (int j = i + 1; j <= dimension; j++) {
                double dx = nodes[i].x - nodes[j].x;
                double dy = nodes[i].y - nodes[j].y;
                double dist = sqrt(dx * dx + dy * dy);
                distances[i][j] = dist;
                distances[j][i] = dist;  // 对称矩阵
            }
        }
    }

    void initializeMatrices() {
        // 使用最近邻启发式计算初始信息素值(贪心算法）
        double totalDist = 0.0;
        vector<bool> visited(dimension + 1, false);
        visited[depot] = true;
        int current = depot;
        int remaining = dimension - 1;

        while (remaining > 0) {
            int nearest = -1;
            double minDist = numeric_limits<double>::max();

            for (int i = 1; i <= dimension; i++) {
                if (!visited[i] && distances[current][i] < minDist) {
                    minDist = distances[current][i];
                    nearest = i;
                }
            }

            if (nearest != -1) {
                totalDist += minDist;
                current = nearest;
                visited[nearest] = true;
                remaining--;
            }
        }

        totalDist += distances[current][depot];  // 返回仓库

        // 初始信息素值
        double initialPheromone = 1.0 / (dimension * totalDist);
        /* double initialPheromone = antCount / (totalDist);*/

        pheromone.resize(dimension + 1, vector<double>(dimension + 1, initialPheromone));
        heuristic.resize(dimension + 1, vector<double>(dimension + 1, 0.0));

        // 预计算启发式信息
        for (int i = 1; i <= dimension; i++) {
            for (int j = 1; j <= dimension; j++) {
                if (i != j) {
                    heuristic[i][j] = 1.0 / distances[i][j];
                }
            }
        }

        // 预计算幂值
        pheromone_power.resize(dimension + 1, vector<double>(dimension + 1, 0.0));
        heuristic_power.resize(dimension + 1, vector<double>(dimension + 1, 0.0));
        updatePowerMatrices();
    }

    // 更新预计算的幂矩阵
    void updatePowerMatrices() {
        for (int i = 1; i <= dimension; i++) {
            for (int j = 1; j <= dimension; j++) {
                if (i != j) {
                    pheromone_power[i][j] = pow(pheromone[i][j], alpha);
                    heuristic_power[i][j] = pow(heuristic[i][j], beta);
                }
            }
        }
    }

    // 添加一个安全的增加评估计数的方法
    bool incrementEvaluation(int count = 1) {
        std::lock_guard<std::mutex> lock(evaluationMutex);
        if (evaluationCount + count >= maxEvaluations) {
            evaluationCount = maxEvaluations;
            return false; // 返回false表示已达到最大评估次数
        }
        evaluationCount += count;
        return true; // 返回true表示可以继续
    }

    // 添加一个检查是否达到最大评估次数的方法
    bool reachedMaxEvaluations() {
        std::lock_guard<std::mutex> lock(evaluationMutex);
        return evaluationCount >= maxEvaluations;
    }

    void solve() {
        auto startTime = chrono::high_resolution_clock::now();


        bestAnt.totalDistance = numeric_limits<double>::max();

        int noImprovementCount = 0;

        // 预留一些评估次数给最后的处理
        const int reservedEvaluations = 100;
        const int effectiveMaxEvaluations = maxEvaluations - reservedEvaluations;

        for (int iter = 0; iter < maxIterations; iter++) {
            // 检查是否接近最大评估次数
            if (evaluationCount >= effectiveMaxEvaluations) {
                cout << "Approaching maximum evaluation count: " << maxEvaluations << endl;
                break;
            }

            // 计算本次迭代最多可用的评估次数
            int availableEvaluations = effectiveMaxEvaluations - evaluationCount;
            int antsToProcess = min(antCount, availableEvaluations);

            if (antsToProcess <= 0) {
                cout << "No more evaluations available" << endl;
                break;
            }

            vector<Ant> ants(antsToProcess);

#pragma omp parallel for
            for (int k = 0; k < antsToProcess; k++) {
                // 在处理每只蚂蚁前检查评估次数
                if (!reachedMaxEvaluations()) {
                    constructSolution(ants[k]);

                    // 增加评估计数
#pragma omp critical

                    // 在进行局部搜索前再次检查评估次数
                    if (!reachedMaxEvaluations() && useLocalSearch) {
                        localSearch(ants[k]);
                    }
                }
            }

            // 如果已达到最大评估次数，跳出循环
            if (reachedMaxEvaluations()) {
                cout << "Reached maximum evaluation count during iteration " << iter << endl;
                break;
            }

            // 找出本次迭代最佳蚂蚁
            if (ants.empty()) break;

            Ant* iterationBest = &ants[0];
            for (int k = 1; k < ants.size(); k++) {
                if (ants[k].totalDistance < iterationBest->totalDistance) {
                    iterationBest = &ants[k];
                }
            }
            currentAnt = *iterationBest;
            // 更新全局最佳解
            bool improved = false;
            if (iterationBest->totalDistance < bestAnt.totalDistance) {
                bestAnt = *iterationBest;
                improved = true;
                noImprovementCount = 0;

                cout << "Iteration " << iter << ": Found better solution, distance = " << bestAnt.totalDistance
                    << ", vehicles = " << bestAnt.usedVehicles
                    << ", evaluations = " << evaluationCount << "/" << maxEvaluations << endl;
            }
            else {
                noImprovementCount++;
            }

            // 更新信息素
            updatePheromone(ants, bestAnt);

            // 输出当前评估次数
            if (iter % 10 == 0) {
                cout << "Iteration " << iter << ": Current evaluations = " << evaluationCount << "/" << maxEvaluations << endl;
            }
        }

        // 输出最终评估次数
        cout << "Total evaluations: " << evaluationCount << endl;

        // 计算总运行时间
        auto endTime = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
        cout << "Total time: " << duration << " seconds" << endl;

        // 输出最终结果
        cout << "\nFinal Results:" << endl;
        cout << "Instance: " << instanceName << endl;
        cout << "Total Distance: " << bestAnt.totalDistance << endl;
        cout << "Vehicles Used: " << bestAnt.usedVehicles << endl;
        cout << "Total Runtime: " << duration << " seconds" << endl;

        // 输出每条路径
        for (int i = 0; i < bestAnt.routes.size(); i++) {
            cout << "Route #" << (i + 1) << ": ";
            for (int j = 0; j < bestAnt.routes[i].size(); j++) {
                cout << bestAnt.routes[i][j];
                if (j < bestAnt.routes[i].size() - 1) {
                    cout << " -> ";
                }
            }
            cout << endl;
        }


    }

    void constructSolution(Ant& ant) {
        vector<bool> visited(dimension + 1, false);
        visited[depot] = true;  // 仓库已访问

        int remainingCustomers = dimension - 1;  // 不包括仓库

        while (remainingCustomers > 0) {
            vector<int> route;
            route.push_back(depot);  // 从仓库开始

            int currentNode = depot;
            int remainingCapacity = capacity;

            while (true) {
                int nextNode = selectNextNode(currentNode, visited, remainingCapacity);

                if (nextNode == -1) {  // 无法继续当前路径
                    route.push_back(depot);  // 返回仓库
                    break;
                }

                route.push_back(nextNode);
                visited[nextNode] = true;
                remainingCapacity -= nodes[nextNode].demand;
                remainingCustomers--;
                currentNode = nextNode;

                // 局部信息素更新
                pheromone[currentNode][nextNode] = (1 - rho) * pheromone[currentNode][nextNode] + rho * (1.0 / (dimension * 100));
                pheromone[nextNode][currentNode] = pheromone[currentNode][nextNode];  // 对称更新

                // 更新预计算的幂值
                pheromone_power[currentNode][nextNode] = pow(pheromone[currentNode][nextNode], alpha);
                pheromone_power[nextNode][currentNode] = pheromone_power[currentNode][nextNode];
            }

            ant.routes.push_back(route);
            if (!incrementEvaluation(1)) {
                return;
            }
        }

        // 计算总距离
        double totalDist = 0.0;
        for (const auto& route : ant.routes) {
            for (int i = 0; i < route.size() - 1; i++) {
                totalDist += distances[route[i]][route[i + 1]];
            }
        }

        ant.totalDistance = totalDist;
        ant.usedVehicles = ant.routes.size();
    }

    int selectNextNode(int currentNode, const vector<bool>& visited, int remainingCapacity) {
        vector<int> candidates;

        // 找出所有可行的候选节点
        for (int i = 1; i <= dimension; i++) {
            if (!visited[i] && nodes[i].demand <= remainingCapacity) {
                candidates.push_back(i);
            }
        }

        if (candidates.empty()) {
            return -1;  // 没有可行的候选节点
        }

        // 使用伪随机比例规则 (Pseudo-Random Proportional Rule)
        double q = dist(rng);

        if (q <= q0) {
            // 贪婪选择
            int bestNode = -1;
            double bestValue = -1.0;

            for (int candidate : candidates) {
                double value = pheromone_power[currentNode][candidate] * heuristic_power[currentNode][candidate];
                if (value > bestValue) {
                    bestValue = value;
                    bestNode = candidate;
                }
            }

            return bestNode;
        }
        else {
            // 随机比例选择
            vector<double> probabilities;
            double denominator = 0.0;

            for (int candidate : candidates) {
                double value = pheromone_power[currentNode][candidate] * heuristic_power[currentNode][candidate];
                probabilities.push_back(value);
                denominator += value;
            }

            // 归一化概率
            for (double& prob : probabilities) {
                prob /= denominator;
            }

            // 轮盘赌选择
            double r = dist(rng);
            double cumulative_prob = 0.0;

            for (int i = 0; i < candidates.size(); i++) {
                cumulative_prob += probabilities[i];
                if (r <= cumulative_prob) {
                    return candidates[i];
                }
            }

            // 如果由于浮点误差没有选择节点，则返回最后一个候选节点
            return candidates.back();
        }
    }

    // 局部搜索优化
    void localSearch(Ant& ant) {
        bool improved = true;
        int iterations = 0;

        // 为局部搜索预留的最大评估次数
        const int maxLocalSearchEvals = 500;
        int localSearchEvals = 0;

        while (improved && iterations < maxLocalSearchIterations && localSearchEvals < maxLocalSearchEvals) {


            localSearchEvals++;
            improved = false;
            iterations++;

            // 2-opt 优化每条路径
            //for (auto& route : ant.routes) {
            //    if (route.size() <= 4) continue;  // 路径太短，不需要优化

            //    for (int i = 1; i < route.size() - 2; i++) {
            //        for (int j = i + 1; j < route.size() - 1; j++) {
            //            // 计算当前路径段的距离
            //           
            //            if (!incrementEvaluation(1)) {
            //                return;
            //            }

            //            double currentDist = distances[route[i - 1]][route[i]] + distances[route[j]][route[j + 1]];

            //            // 计算交换后的距离
            //            double newDist = distances[route[i - 1]][route[j]] + distances[route[i]][route[j + 1]];

            //            if (newDist < currentDist) {
            //                // 反转子路径
            //                reverse(route.begin() + i, route.begin() + j + 1);
            //                improved = true;
            //            }
            //        }
            //    }
            //}

            // 尝试在路径之间移动客户
            if (ant.routes.size() > 1) {
                for (int r1 = 0; r1 < ant.routes.size(); r1++) {
                    for (int r2 = 0; r2 < ant.routes.size(); r2++) {
                        if (r1 == r2) continue;

                        auto& route1 = ant.routes[r1];
                        auto& route2 = ant.routes[r2];

                        // 计算路径2的剩余容量
                        int route2RemainingCapacity = capacity;
                        for (int i = 1; i < route2.size() - 1; i++) {
                            route2RemainingCapacity -= nodes[route2[i]].demand;
                        }

                        // 尝试从路径1移动客户到路径2
                        for (int i = 1; i < route1.size() - 1; i++) {
                            int customer = route1[i];

                            // 检查容量约束
                            if (nodes[customer].demand > route2RemainingCapacity) continue;

                            // 计算从路径1移除客户的节省
                            double savingRoute1 = distances[route1[i - 1]][route1[i]] + distances[route1[i]][route1[i + 1]] - distances[route1[i - 1]][route1[i + 1]];

                            // 尝试将客户插入路径2的每个可能位置
                            for (int j = 1; j < route2.size(); j++) {

                                double costRoute2 = distances[route2[j - 1]][customer] + distances[customer][route2[j]] - distances[route2[j - 1]][route2[j]];

                                if (savingRoute1 > costRoute2) {
                                    // 移动客户
                                    route2.insert(route2.begin() + j, customer);
                                    route1.erase(route1.begin() + i);

                                    // 更新路径2的剩余容量
                                    route2RemainingCapacity -= nodes[customer].demand;

                                    if (!incrementEvaluation(1)) {
                                        return;
                                    }
                                    improved = true;
                                    break;
                                }
                            }

                            if (improved) break;
                        }

                        if (improved) break;
                    }

                    if (improved) break;
                }
            }

            // 如果有改进，重新计算总距离
            if (improved) {
                double totalDist = 0.0;
                for (const auto& route : ant.routes) {
                    for (int i = 0; i < route.size() - 1; i++) {
                        totalDist += distances[route[i]][route[i + 1]];
                    }
                }
                ant.totalDistance = totalDist;
            }

            // 在每次重要操作后检查评估次数
            if (reachedMaxEvaluations()) {
                return;
            }
        }

        // 移除空路径
        ant.routes.erase(
            remove_if(ant.routes.begin(), ant.routes.end(),
                [](const vector<int>& route) { return route.size() <= 2; }),
            ant.routes.end()
        );

        ant.usedVehicles = ant.routes.size();
    }

    void updatePheromone(const vector<Ant>& ants, const Ant& bestAnt) {
        // 信息素蒸发
        for (int i = 1; i <= dimension; i++) {
            for (int j = i + 1; j <= dimension; j++) {
                pheromone[i][j] *= (1 - rho);
                pheromone[j][i] = pheromone[i][j];  // 对称更新
            }
        }

        // 全局最优蚂蚁留下信息素
        double deltaTau = 1.0 / bestAnt.totalDistance;

        for (const auto& route : bestAnt.routes) {
            for (int i = 0; i < route.size() - 1; i++) {
                int from = route[i];
                int to = route[i + 1];
                pheromone[from][to] += deltaTau;
                pheromone[to][from] = pheromone[from][to];  // 对称更新
            }
        }

        // 更新预计算的幂矩阵
        updatePowerMatrices();
    }


};

int main() {
    // 手动列出所有.vrp文件路径
    std::vector<string> instances = {
     "A-n32-k5.vrp",
     "A-n33-k5.vrp",
     "A-n33-k6.vrp",
     "A-n34-k5.vrp",
     "A-n36-k5.vrp",
     "A-n37-k5.vrp",
     "A-n37-k6.vrp",
     "A-n38-k5.vrp",
     "A-n39-k5.vrp",
     "A-n39-k6.vrp",
     "A-n44-k6.vrp",
     "A-n45-k6.vrp",
     "A-n45-k7.vrp",
     "A-n46-k7.vrp",
     "A-n48-k7.vrp",
     "A-n53-k7.vrp",
     "A-n54-k7.vrp",
     "A-n55-k9.vrp",
      "A-n60-k9.vrp",
      "A-n61-k9.vrp",
      "A-n62-k8.vrp",
      "A-n63-k9.vrp",
      "A-n63-k10.vrp",
      "A-n64-k9.vrp",
     "A-n65-k9.vrp",
     "A-n69-k9.vrp",
     "A-n80-k10.vrp",
    };
    // 打开CSV文件
    std::ofstream csv("蚁群.csv");

    // 写入CSV文件头
    csv << "Instance";
    for (int run = 1; run <= 25; ++run) {
        csv << ",Run " << run << " Cost";
    }
    csv << ",Average Cost,Average Time (s)\n";

    // 顺序执行所有实例
    for (const auto& instance : instances) {
        double totalCost = 0;
        double totalTime = 0;
      

        // 写入实例名称
        csv << instance;

        // 运行25次
        for (int run = 0; run < 25; ++run) {
            CVRP cvrp;

            // 读取问题数据
            if (!cvrp.readInstance(instance)) {
                std::cerr << "Failed to read input file: " << instance << std::endl;
                continue;
            }

            // 计时开始
            auto startTime = std::chrono::high_resolution_clock::now();

            // 运行算法
            cvrp.solve();

            // 计时结束
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            double timeSeconds = duration / 1000.0;

            // 获取当前运行的结果
            double currentCost = cvrp.bestAnt.totalDistance; // 获取当前解的成本
            totalCost += currentCost;
            totalTime += timeSeconds;



            // 输出当前运行的结果
            std::cout << "Run " << run + 1 << " for " << instance
                << " Cost: " << currentCost
                << " Time: " << timeSeconds << "s" << std::endl;

            // 将当前运行的结果写入CSV文件
            csv << "," << currentCost;
        }
        double avgCost = totalCost / 25;
        double avgTime = totalTime / 25;

        // 将平均值写入CSV文件
        csv << "," << avgCost << "," << avgTime << "\n";

        // 输出实例的平均结果
        std::cout << "Finished: " << instance
            << " Avg Cost: " << avgCost
            << " Avg Time: " << avgTime << "s" << std::endl;



    }

    csv.close();
    return 0;
}