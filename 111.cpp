#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <limits>
#include <ctime>
#include <numeric>
#include <string>
#include <numeric>
#include <sstream>
using namespace std;

struct Node {
    double x, y;
    int demand;
};

//// 算法参数
const int MaxFEs = 50000;
const double local_phi = 0.1;

const int num_ants = 50;        // 增加蚂蚁数量
const double alpha = 0.8;       // 降低信息素权重
const double beta = 3.5;        // 降低启发式影响，增强搜索
const double global_rho = 0.25; // 增强全局信息素更新
const double q0 = 0.75;         // 增加随机探索
// 配合局部搜索参数
const int max_2opt_iter = 1;    // 控制优化深度
// 问题相关变量
int capacity, n;
vector<Node> nodes;
vector<vector<double>> dist;
vector<vector<double>> pheromone;
double tau0;
int FEs = 0;

// 最优解存储
vector<vector<int>> best_solution;
double best_cost = numeric_limits<double>::max();

// 计算欧氏距离
double euclideanDistance(const Node& a, const Node& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}


// 辅助函数：去除字符串头尾空白
string trim(const string& s) {
    auto start = s.begin();
    while (start != s.end() && isspace(*start)) start++;
    auto end = s.end();
    do { end--; } while (distance(start, end) > 0 && isspace(*end));
    return string(start, end + 1);
}

// 读取.vrp文件
void readData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    bool reading_nodes = false;
    bool reading_demands = false;
    bool reading_depot = false;
    int depot_id = 0;

    while (getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (line.find("CAPACITY") != string::npos) {
            sscanf_s(line.c_str(), "CAPACITY : %d", &capacity);
        }
        else if (line.find("DIMENSION") != string::npos) {
            sscanf_s(line.c_str(), "DIMENSION : %d", &n);
            nodes.resize(n);
        }
        else if (line == "NODE_COORD_SECTION") {
            reading_nodes = true;
            continue;
        }
        else if (line == "DEMAND_SECTION") {
            reading_demands = true;
            reading_nodes = false;
            continue;
        }
        else if (line == "DEPOT_SECTION") {
            reading_demands = false;
            reading_depot = true;
            continue;
        }
        else if (line == "EOF") break;

        if (reading_nodes) {
            int id;
            double x, y;
            stringstream ss(line);
            ss >> id >> x >> y;
            if (id > 0 && id <= n) {
                nodes[id - 1].x = x;
                nodes[id - 1].y = y;
            }
        }
        else if (reading_demands) {
            int id, demand;
            stringstream ss(line);
            ss >> id >> demand;
            if (id > 0 && id <= n) {
                nodes[id - 1].demand = demand;
            }
        }
        else if (reading_depot) {
            int d;
            stringstream ss(line);
            while (ss >> d) {
                if (d == -1) break;
                depot_id = d;
            }
        }
    }

    // 验证仓库需求是否为0
    if (depot_id > 0 && depot_id <= n) {
        nodes[depot_id - 1].demand = 0;
    }
}
// 初始化距离矩阵
void initDistanceMatrix() {
    dist.resize(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dist[i][j] = euclideanDistance(nodes[i], nodes[j]);
        }
    }
}

// 初始化信息素矩阵
void initPheromone() {
    // 计算最近邻启发式解
    vector<bool> visited(n, false);
    visited[0] = true;
    double Lnn = 0;
    int current = 0, count = 1;

    while (count < n) {
        double min_dist = numeric_limits<double>::max();
        int next = -1;
        for (int i = 0; i < n; ++i) {
            if (!visited[i] && dist[current][i] < min_dist) {
                min_dist = dist[current][i];
                next = i;
            }
        }
        if (next == -1) break;
        Lnn += min_dist;
        current = next;
        visited[next] = true;
        count++;
    }
    Lnn += dist[current][0];

    tau0 = 1.0 / (n * Lnn);
    pheromone.resize(n, vector<double>(n, tau0));
}
// 计算路径总长度
double calculateCost(const vector<vector<int>>& routes) {
    double cost = 0;
    for (const auto& route : routes) {
        for (size_t i = 0; i < route.size() - 1; ++i) {
            cost += dist[route[i]][route[i + 1]];
        }
    }
    return cost;
}

// 2-opt 交换：优化单条路径
void apply2Opt(vector<int>& route) {
    bool improved = true;



    int iter_count = 0;
    while (improved && iter_count++ < max_2opt_iter && FEs < MaxFEs) {  // 增加终止条件检查
        improved = false;
        int route_size = route.size();

        for (int i = 1; i < route_size - 2; ++i) {
            for (int j = i + 1; j < route_size - 1; ++j) {
                // 计算旧距离
                double old_dist = dist[route[i - 1]][route[i]] + dist[route[j]][route[j + 1]];

                // 计算新距离（需要评估整个路径）
                vector<int> new_route = route;
                reverse(new_route.begin() + i, new_route.begin() + j + 1);
                double new_dist = calculateCost({ new_route });  // 评估新解
                FEs++;  // 增加评估计数

                if (new_dist < old_dist && FEs < MaxFEs) {
                    route = new_route;
                    improved = true;
                }
            }
        }
    }
}

// 构建蚂蚁路径
vector<vector<int>> constructSolution() {
    vector<vector<int>> routes;
    vector<bool> visited(n, false);
    visited[0] = true;
    int remaining = n - 1;

    while (remaining > 0) {
        vector<int> route = { 0 };
        int load = 0, current = 0;

        while (true) {
            vector<int> candidates;
            for (int i = 1; i < n; ++i) {
                if (!visited[i] && nodes[i].demand <= (capacity - load)) {
                    candidates.push_back(i);
                }
            }
            if (candidates.empty()) break;

            // ACS状态转移规则
            double max_product = -1;
            int selected = -1;
            vector<double> probabilities;
            double sum = 0.0;

            for (int i : candidates) {
                double phe = pheromone[current][i];
                double eta = 1.0 / dist[current][i];
                double product = pow(phe, alpha) * pow(eta, beta);
                probabilities.push_back(product);
                sum += product;
            }

            double q = (double)rand() / RAND_MAX;
            if (q <= q0) {
                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (probabilities[i] > max_product) {
                        max_product = probabilities[i];
                        selected = candidates[i];
                    }
                }
            }
            else {
                double r = (double)rand() / RAND_MAX * sum;
                double accum = 0;
                for (size_t i = 0; i < candidates.size(); ++i) {
                    accum += probabilities[i];
                    if (accum >= r) {
                        selected = candidates[i];
                        break;
                    }
                }
            }

            if (selected == -1) break;

            // 更新路径状态
            route.push_back(selected);
            load += nodes[selected].demand;
            visited[selected] = true;
            remaining--;
            current = selected;

            // 局部信息素更新
            pheromone[current][selected] = (1 - local_phi) * pheromone[current][selected] + local_phi * tau0;
        }

        route.push_back(0);

        //apply2Opt(route);
        routes.push_back(route);
    }
    return routes;
}





// 全局信息素更新
void globalUpdate() {
    double delta = 1.0 / best_cost;
    for (const auto& route : best_solution) {
        for (size_t i = 0; i < route.size() - 1; ++i) {
            int from = route[i], to = route[i + 1];
            pheromone[from][to] = (1 - global_rho) * pheromone[from][to] + global_rho * delta;
        }
    }
}

int main() {
    srand(time(0));
    readData("A-n34-k5.vrp");
    initDistanceMatrix();
    initPheromone();

    while (FEs < MaxFEs) {
        vector<vector<int>> iteration_best;
        double iteration_cost = numeric_limits<double>::max();

        // 蚂蚁搜索
        for (int ant = 0; ant < num_ants && FEs < MaxFEs; ++ant) {
            auto solution = constructSolution();
            double cost = calculateCost(solution);
            FEs++;

            if (cost < iteration_cost) {
                iteration_cost = cost;
                iteration_best = solution;
            }

            if (cost < best_cost) {
                best_cost = cost;
                best_solution = solution;
            }
        }

        // 全局信息素更新
        if (FEs < MaxFEs&&!best_solution.empty()) {
            globalUpdate();
        }

        cout << "FEs: " << FEs << "/" << MaxFEs
            << " Best: " << best_cost << endl;
    }

    cout << "\nOptimal Cost: " << best_cost << endl;
    cout << "Routes:\n";
    for (const auto& route : best_solution) {
        for (int node : route) cout << node << " ";
        cout << endl;
    }
    return 0;
}