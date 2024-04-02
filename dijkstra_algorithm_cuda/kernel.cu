#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <limits.h>
#include <chrono>
#include <vector>
#include <queue>
#include <random>
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 256

void printResult(const vector<int>& dist, int N) {
    for (int i = 0; i < N; i++) {
        cout << i << "\t \t" << dist[i] << endl;
    }
}

__global__ void dijkstra(float* d, int* p, int* visited, int n, int start)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n && !visited[i])
    {
        visited[i] = 1;

        for (int j = 0; j < n; j++)
        {
            int weight = p[i * n + j];
            if (weight != 0 && d[i] + weight < d[j])
            {
                d[j] = d[i] + weight;
            }

            // Check the reverse direction for unidirectional edges
            weight = p[j * n + i];
            if (weight != 0 && d[j] + weight < d[i])
            {
                d[i] = d[j] + weight;
            }
        }
    }
}

void parallel_dijkstra_gpu(std::vector<std::vector<int>>& graph, int start)
{
    int n = graph.size();
    float* d = new float[n];
    int* visited = new int[n];
    std::fill(d, d + n, std::numeric_limits<float>::infinity());
    std::fill(visited, visited + n, 0);
    d[start] = 0;
    visited[start] = 1;

    float* d_gpu;
    cudaMalloc(&d_gpu, n * sizeof(float));
    cudaMemcpy(d_gpu, d, n * sizeof(float), cudaMemcpyHostToDevice);

    int* p_gpu;
    cudaMalloc(&p_gpu, n * n * sizeof(int));
    int* p_host = new int[n * n];
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            p_host[i * n + j] = graph[i][j];
        }
    }
    cudaMemcpy(p_gpu, p_host, n * n * sizeof(int), cudaMemcpyHostToDevice);
    delete[] p_host;

    int* visited_gpu;
    cudaMalloc(&visited_gpu, n * sizeof(int));
    cudaMemcpy(visited_gpu, visited, n * sizeof(int), cudaMemcpyHostToDevice);

    auto start_time = high_resolution_clock::now();

    dijkstra << <(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_gpu, p_gpu, visited_gpu, n, start);
    cudaDeviceSynchronize();

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);

    cout << "Execution time (GPU): " << duration.count() << " microseconds" << endl;

    cudaMemcpy(d, d_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
    {
        std::cout << i << "\t \t" << d[i] << std::endl;
    }

    cudaFree(d_gpu);
    cudaFree(p_gpu);
    cudaFree(visited_gpu);
    delete[] d;
    delete[] visited;
}

std::vector<std::vector<int>> generate_complex_graph(int n, int min_weight = 1, int max_weight = 10)
{
    std::vector<std::vector<int>> graph(n, std::vector<int>(n, 0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_weight, max_weight);

    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            int weight = dis(gen);
            graph[i][j] = weight;
            graph[j][i] = weight;
        }
    }

    return graph;
}

//dijkstra on cpu
void dijkstraCPU(const vector<vector<int>>& graph, int src, int N) {
    vector<int> dist(N, INT_MAX);
    vector<bool> visited(N, false);
    dist[src] = 0;

    auto start = high_resolution_clock::now();

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({ 0, src });

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;

        for (int v = 0; v < N; v++) {
            if (!visited[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                pq.push({ dist[v], v });
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Execution time (CPU): " << duration.count() << " microseconds" << endl;

    printResult(dist, N);
}

int main()
{
    int n = 5; // Number of nodes
    int start = 0; // Starting node
    auto graph = generate_complex_graph(n);
    vector<vector<int>> graph1 = {
      {0, 1, 4, 0, 0},
      {0, 0, 2, 5, 12},
      {0, 0, 0, 2, 0},
      {0, 0, 0, 0, 3},
      {0, 0, 0, 0, 0},
    };

    std::cout << "Graph:" << std::endl;
    for (const auto& row : graph)
    {
        for (int weight : row)
        {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
    }

    parallel_dijkstra_gpu(graph, start);
    dijkstraCPU(graph, start, n);

    return 0;
}