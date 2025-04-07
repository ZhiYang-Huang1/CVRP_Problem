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

    // 添加评估计数器
    int evaluationCount;
    int maxEvaluations;

public:
    CVRP() : rng(chrono::high_resolution_clock::now().time_since_epoch().count()), dist(0.0, 1.0) {
        antCount = 25;                // 减少蚂蚁数量，提高效率
        maxIterations = 1000;
        alpha = 1.0;
        beta = 2.5;                   // 增加启发式信息权重
        rho = 0.1;
        q0 = 0.9;                     // 增加利用率
        depot = 1;
        useLocalSearch = true;        // 启用局部搜索
        maxLocalSearchIterations = 100;

        // 初始化评估计数器
        evaluationCount = 0;
        maxEvaluations = 50000;
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
        // 使用最近邻启发式计算初始信息素值
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
        //double initialPheromone = 1.0 / (dimension * totalDist);
        double initialPheromone = antCount / (totalDist);

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

    void solve() {
        auto startTime = chrono::high_resolution_clock::now();

        Ant bestAnt;
        bestAnt.totalDistance = numeric_limits<double>::max();

        // 迭代计数器和无改进计数器
        int noImprovementCount = 0;

        for (int iter = 0; iter < maxIterations; iter++) {
            // 检查是否达到最大评估次数
            if (evaluationCount >= maxEvaluations) {
                cout << "Reached maximum evaluation count: " << maxEvaluations << endl;
                break;
            }

            vector<Ant> ants(antCount);

            // 并行构建解决方案
#pragma omp parallel for
            for (int k = 0; k < antCount; k++) {
                constructSolution(ants[k]);

                // 增加评估计数
#pragma omp atomic
                evaluationCount += 1;

                // 局部搜索优化
                if (useLocalSearch) {
                    localSearch(ants[k]);
                }
            }

            // 找出本次迭代最佳蚂蚁
            Ant* iterationBest = &ants[0];
            for (int k = 1; k < antCount; k++) {
                if (ants[k].totalDistance < iterationBest->totalDistance) {
                    iterationBest = &ants[k];
                }
            }

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

            // 每100次迭代输出当前最佳解
            if (iter % 100 == 0) {
                cout << "Iteration " << iter << ": Current best distance = " << bestAnt.totalDistance
                    << ", vehicles = " << bestAnt.usedVehicles << endl;

                // 计算并显示运行时间
                auto currentTime = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
                cout << "Time elapsed: " << duration << " seconds" << endl;
            }

            // 如果长时间没有改进，动态调整参数
            if (noImprovementCount > 200) {
                q0 = max(0.5, q0 - 0.05);  // 增加探索
                noImprovementCount = 0;
                cout << "No improvement for 200 iterations, adjusting q0 to " << q0 << endl;
            }
        }

        // 计算总运行时间
        auto endTime = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();

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

        // 输出最终评估次数
        cout << "Total evaluations: " << evaluationCount << endl;

        // 保存结果到文件
        saveResult(bestAnt);
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
                pheromone[currentNode][nextNode] = (1 - rho) * pheromone[currentNode][nextNode] + rho * (1.0 / (dimension * 10));
                pheromone[nextNode][currentNode] = pheromone[currentNode][nextNode];  // 对称更新

                // 更新预计算的幂值
                pheromone_power[currentNode][nextNode] = pow(pheromone[currentNode][nextNode], alpha);
                pheromone_power[nextNode][currentNode] = pheromone_power[currentNode][nextNode];
            }

            ant.routes.push_back(route);
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

        while (improved && iterations < maxLocalSearchIterations) {
            // 检查是否达到最大评估次数
            if (evaluationCount >= maxEvaluations) {
                return;
            }

            improved = false;
            iterations++;

            // 增加评估计数
            evaluationCount += 1;

            // 2-opt 优化每条路径
            for (auto& route : ant.routes) {
                if (route.size() <= 4) continue;  // 路径太短，不需要优化

                for (int i = 1; i < route.size() - 2; i++) {
                    for (int j = i + 1; j < route.size() - 1; j++) {
                        // 计算当前路径段的距离
                        double currentDist = distances[route[i - 1]][route[i]] + distances[route[j]][route[j + 1]];

                        // 计算交换后的距离
                        double newDist = distances[route[i - 1]][route[j]] + distances[route[i]][route[j + 1]];

                        if (newDist < currentDist) {
                            // 反转子路径
                            reverse(route.begin() + i, route.begin() + j + 1);
                            improved = true;
                        }
                    }
                }
            }

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

    void saveResult(const Ant& ant) {
        size_t pos = instanceName.find_last_of("/\\");
        string name = (pos == string::npos) ? instanceName : instanceName.substr(pos + 1);
        name = name.substr(0, name.find_last_of('.'));

        string filename = name + ".sol";
        ofstream file(filename);

        if (!file.is_open()) {
            cerr << "Cannot create result file: " << filename << endl;
            return;
        }

        for (int i = 0; i < ant.routes.size(); i++) {
            file << "Route #" << (i + 1) << ":";
            // 不输出起点和终点的仓库
            for (int j = 1; j < ant.routes[i].size() - 1; j++) {
                file << " " << ant.routes[i][j];
            }
            file << endl;
        }

        file << "Cost " << fixed << setprecision(0) << ant.totalDistance << endl;
        file.close();

        cout << "Results saved to: " << filename << endl;
    }
};

int main() {
    // 在这里手动指定要运行的实例文件
    string instanceFile = "A-n33-k5.vrp";  // 修改这里来运行不同的实例

    cout << "Processing instance: " << instanceFile << endl;

    CVRP cvrp;
    if (cvrp.readInstance(instanceFile)) {
        cvrp.solve();
    }
    else {
        cout << "Cannot read instance file, please check if the path is correct" << endl;
    }

    cout << "Press any key to continue..." << endl;
    getchar();

    return 0;
}