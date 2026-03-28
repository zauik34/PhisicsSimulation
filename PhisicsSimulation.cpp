#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>

#include <SDL3/SDL.h>

const int DOTS_COUNT = 1000;
const float eps = 1000.0f;
const float G = 0.5f;
const float bounceForce = -0.8f;

const float width = 1000;
const float height = 1000;
const int numThreads = 16;

struct Particle {
    float x, y;
    float vx, vy;
    float m;
};
struct QuadtreeNode {
    float xMin, xMax, yMin, yMax;

    float centerX = 0.0f, centerY = 0.0f;
    float totalMass = 0.0f;

    Particle* p = nullptr;

    QuadtreeNode* children[4] = { nullptr, nullptr, nullptr, nullptr };

    QuadtreeNode(float x1, float x2, float y1, float y2)
        : xMin(x1), xMax(x2), yMin(y1), yMax(y2) {
    }
};

class Quadtree {
public:
    QuadtreeNode* root;
    QuadtreeNode* pool;
    int poolPtr = 0;
    const int MAX_NODES = DOTS_COUNT * 8;
    Quadtree(float minX, float maxX, float minY, float maxY) {
        pool = (QuadtreeNode*)malloc(sizeof(QuadtreeNode) * MAX_NODES);
        root = new(&pool[0]) QuadtreeNode(minX, maxX, minY, maxY);
        poolPtr = 1;
    }

    void insert(QuadtreeNode* node, Particle* newP, int depth = 0) {
        if (depth > 18) {
            node->p = newP;
            return;
        }
        if (node->p == nullptr && node->children[0] == nullptr) {
            node->p = newP;
            return;
        }
        if (node->children[0] == nullptr) {
            Particle* oldP = node->p;
            node->p = nullptr;

            subdivide(node);

            insert(node->children[getQuarter(node, oldP->x, oldP->y)], oldP, depth + 1);
        }

        insert(node->children[getQuarter(node, newP->x, newP->y)], newP, depth + 1);
    }
    int getQuarter(QuadtreeNode* node, float px, float py) {
        float midX = (node->xMin + node->xMax) * 0.5f;
        float midY = (node->yMin + node->yMax) * 0.5f;

        int index = 0;
        if (px >= midX) index |= 1;
        if (py >= midY) index |= 2;

        return index;
    }
    void subdivide(QuadtreeNode* node) {
        float midX = (node->xMin + node->xMax) * 0.5f;
        float midY = (node->yMin + node->yMax) * 0.5f;

        node->children[0] = new(&pool[poolPtr++]) QuadtreeNode(node->xMin, midX, node->yMin, midY);
        node->children[1] = new(&pool[poolPtr++]) QuadtreeNode(midX, node->xMax, node->yMin, midY);
        node->children[2] = new(&pool[poolPtr++]) QuadtreeNode(node->xMin, midX, midY, node->yMax);
        node->children[3] = new(&pool[poolPtr++]) QuadtreeNode(midX, node->xMax, midY, node->yMax);
    }

    void updateMassProperties(QuadtreeNode* node) {
        if (!node) return;
        if (node->p != nullptr) {
            node->totalMass = node->p->m;
            node->centerX = node->p->x;
            node->centerY = node->p->y;
        }
        else {
            node->totalMass = 0;
            node->centerX = 0;
            node->centerY = 0;

            for (int i = 0; i < 4; i++) {
                if (node->children[i] != nullptr) {
                    updateMassProperties(node->children[i]);

                    float m = node->children[i]->totalMass;
                    node->totalMass += m;

                    node->centerX += node->children[i]->centerX * m;
                    node->centerY += node->children[i]->centerY * m;
                }
            }
            if (node->totalMass > 0) {
                node->centerX /= node->totalMass;
                node->centerY /= node->totalMass;
            }
        }
    }
    void applyGravity(float m1, float m2, float dx, float dy, float r2, float& fx, float& fy) {
        float r = sqrtf(r2);
        float invR3 = 1.0f / (r * r2);
        float force = G * m1 * m2 * invR3;
        fx += force * dx;
        fy += force * dy;
    }
    void calculateForce(Particle* p, QuadtreeNode* node, float& totalFx, float& totalFy) {
        if (!node) return;
        float dx = node->centerX - p->x;
        float dy = node->centerY - p->y;
        float r2 = dx * dx + dy * dy + eps;
        float r = sqrtf(r2);

        float s = node->xMax - node->xMin;
        float theta = 0.6f;

        if (node->p != nullptr) {
            if (node->p != p) {
                applyGravity(p->m, node->totalMass, dx, dy, r2, totalFx, totalFy);
            }
        }
        else if (s / r < theta) {
            applyGravity(p->m, node->totalMass, dx, dy, r2, totalFx, totalFy);
        }
        else {
            for (int i = 0; i < 4; i++) {
                calculateForce(p, node->children[i], totalFx, totalFy);
            }
        }
    }

    void reset(float minX, float maxX, float minY, float maxY) {
        poolPtr = 0;
        root = new(&pool[poolPtr++]) QuadtreeNode(minX, maxX, minY, maxY);
    }
};

float deltaTime;
Uint64 previousTime = 0;

void calculateDeltaTime() {
    Uint64 currentTime = SDL_GetTicks();
    deltaTime = (currentTime - previousTime) / 1000.0f;
    previousTime = currentTime;
    if (deltaTime > 0.1f) deltaTime = 0.1f;
}

std::mutex mtx;
std::condition_variable cv;
std::atomic<int> current_frame{ 0 };
std::atomic<bool> running{ true };

std::mutex mtx_main;
std::condition_variable cv_main;
std::atomic<int> completed_tasks{ 0 };

void updatePhisics(std::vector<Particle>& particles, int start, int end, Quadtree& tree) {
    for (int i = start; i < end; i++) {
        float totalFx = 0, totalFy = 0;

        tree.calculateForce(&particles[i], tree.root, totalFx, totalFy);

        float invMi = 1.0f / particles[i].m;
        particles[i].vx += (totalFx * invMi) * deltaTime;
        particles[i].vy += (totalFy * invMi) * deltaTime;
    }
}
void worker(int start, int end, std::vector<Particle>& particles, Quadtree& tree) {
    int last_frame = 0;
    while (running) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return current_frame > last_frame || !running; });
        }
        if (!running) break;

        last_frame = current_frame;

        updatePhisics(particles, start, end, tree);

        {
            std::lock_guard<std::mutex> lock(mtx_main);
            completed_tasks++;
        }
        cv_main.notify_all();
    }
}

int main(int argc, char* argv[]) {
    Quadtree tree(-width, width, -height, height);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> posX(0, width);
    std::uniform_real_distribution<float> posY(0, height);
    std::uniform_real_distribution<float> velocity(-20, 20);
    std::uniform_real_distribution<float> mass(10, 100);

    std::vector<Particle> particles(DOTS_COUNT);
    std::vector<SDL_FPoint> points(DOTS_COUNT);
    std::vector<SDL_Vertex> vertices(DOTS_COUNT * 3);

    float centerX = width / 2.0f;
    float centerY = height / 2.0f;

    for (auto& p : particles) {
        p.x = posX(gen);
        p.y = posY(gen);
        p.m = mass(gen);

        float dx = p.x - centerX;
        float dy = p.y - centerY;
        float dist = sqrtf(dx * dx + dy * dy) + 0.1f;

        float orbitalSpeed = 15.0f;

        p.vx = -(dy / dist) * orbitalSpeed;
        p.vy = (dx / dist) * orbitalSpeed;
        //p.vx = 0;
        //p.vy = 0;
    }


    if (!SDL_Init(SDL_INIT_VIDEO)) return -1;
    SDL_Window* window = SDL_CreateWindow("PhisicsDots", (int)width, (int)height, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    //SDL_SetRenderVSync(renderer, 1);

    std::vector<std::thread> threads;
    int chunkSize = DOTS_COUNT / numThreads;
    for (int i = 0; i < numThreads; i++) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? DOTS_COUNT : start + chunkSize;
        threads.push_back(std::thread(worker, start, end, std::ref(particles), std::ref(tree)));
    }

    SDL_Event event;
    previousTime = SDL_GetTicks();
    Uint64 lastFpsUpdateTime = SDL_GetTicks();
    int frameCount = 0;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) running = false;
        }
        float minX = particles[0].x, maxX = particles[0].x;
        float minY = particles[0].y, maxY = particles[0].y;

        for (const auto& p : particles) {
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        minX = std::max(-5000.0f, minX);
        maxX = std::min(5000.0f, maxX);
        minY = std::max(-5000.0f, minY);
        maxY = std::min(5000.0f, maxY);

        tree.reset(minX, maxX, minY, maxY);
        for (int i = 0; i < DOTS_COUNT; i++) {
            if (particles[i].x >= minX && particles[i].x <= maxX &&
                particles[i].y >= minY && particles[i].y <= maxY) {
                tree.insert(tree.root, &particles[i]);
            }
        }
        tree.updateMassProperties(tree.root);

        completed_tasks = 0;
        {
            std::lock_guard<std::mutex> lock(mtx);
            current_frame++;
        }
        cv.notify_all();

        {
            std::unique_lock<std::mutex> main_lock(mtx_main);
            cv_main.wait(main_lock, [] { return completed_tasks == numThreads; });
        }

        for (auto& p : particles) {
            p.x += p.vx * deltaTime;
            p.y += p.vy * deltaTime;
        }
        for (int i = 0; i < DOTS_COUNT; i++) {
            int idx = i * 3;
            float x = particles[i].x;
            float y = particles[i].y;

            float speed = sqrtf(particles[i].vx * particles[i].vx + particles[i].vy * particles[i].vy);

            float r = std::min(255.0f, speed * 10.0f) / 255.0f;
            float b = (255.0f - (r * 255.0f)) / 255.0f;
            float g = 100.0f / 255.0f;

            SDL_FColor col = { r, g, b, 1.0f };

            vertices[idx].position = { x, y - 1 };
            vertices[idx + 1].position = { x - 1, y + 1 };
            vertices[idx + 2].position = { x + 1, y + 1 };

            vertices[idx].color = vertices[idx + 1].color = vertices[idx + 2].color = col;
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderGeometry(renderer, nullptr, vertices.data(), (int)vertices.size(), nullptr, 0);
        SDL_RenderPresent(renderer);

        calculateDeltaTime();
        frameCount++;
        Uint64 currentTime = SDL_GetTicks();

        if (currentTime - lastFpsUpdateTime >= 1000) {
            float fps = frameCount / ((currentTime - lastFpsUpdateTime) / 1000.0f);

            std::string title = "PhisicsDots | FPS: " + std::to_string((int)fps) + " | Stars: " + std::to_string(DOTS_COUNT);

            SDL_SetWindowTitle(window, title.c_str());

            frameCount = 0;
            lastFpsUpdateTime = currentTime;
        }
    }

    running = false;
    current_frame++;
    cv.notify_all();
    for (auto& t : threads) if (t.joinable()) t.join();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}