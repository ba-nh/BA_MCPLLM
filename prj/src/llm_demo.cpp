#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <csignal>
#include <curl/curl.h>
#include <json/json.h>
#include <cstring>
#include <thread>
#include "rkllm.h"

using namespace std;

LLMHandle llmHandle = nullptr;
std::string g_prompt_str;

double rag_search_time = 0.0, prompt_gen_time = 0.0, llm_infer_time = 0.0, total_time = 0.0;
auto response_start_time = chrono::high_resolution_clock::now();
auto llm_start_time = chrono::high_resolution_clock::now();

const string MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0.rkllm";
const string BASE_PATH = "/home/odroid/Desktop/BAMCP/";
const string RAG_SERVER_URL = "http://localhost:5000/rag";
const string FULL_MODEL_PATH = BASE_PATH + MODEL_NAME;

const int PROMPT_CHAR_LIMIT = 3000;

size_t WriteCallback(void* contents, size_t size, size_t nmemb, string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

string callRAGAPI(const string& query) {
    CURL* curl = curl_easy_init();
    string response;

    if (curl) {
        Json::Value requestData;
        requestData["query"] = query;
        requestData["k"] = 2;
        requestData["alpha"] = 0.5;

        Json::FastWriter writer;
        string jsonData = writer.write(requestData);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, RAG_SERVER_URL.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        if (curl_easy_perform(curl) != CURLE_OK) response.clear();

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    return response;
}

void exit_handler(int signal) {
    if (llmHandle) rkllm_destroy(llmHandle);
    exit(signal);
}

void callback(RKLLMResult* result, void*, LLMCallState state) {
    static std::string result_accumulator;

    switch (state) {
        case RKLLM_RUN_NORMAL:
            if (result && result->text) {
                fwrite(result->text, sizeof(char), strlen(result->text), stdout);
                fflush(stdout);
                result_accumulator += result->text;
            }
            break;

        case RKLLM_RUN_FINISH:
            llm_infer_time = chrono::duration<double>(chrono::high_resolution_clock::now() - llm_start_time).count();
            total_time = chrono::duration<double>(chrono::high_resolution_clock::now() - response_start_time).count();
            printf("\n[RAG: %.3fs] [Prompt: %.3fs] [LLM: %.3fs] [Total: %.3fs]\n",
                   rag_search_time, prompt_gen_time, llm_infer_time, total_time);
            result_accumulator.clear();
            break;

        case RKLLM_RUN_ERROR:
            printf("[Error] rkllm_run failed.\n");
            break;

        default:
            break;
    }
}

int main() {
    signal(SIGINT, exit_handler);
    curl_global_init(CURL_GLOBAL_ALL);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = FULL_MODEL_PATH.c_str();
    param.skip_special_token = true;
    param.is_async = false;
    param.temperature = 0.2;
    param.top_p = 0.8;
    param.top_k = 40;
    param.repeat_penalty = 1.2f;
    param.frequency_penalty = 0.1f;
    param.presence_penalty = 0.1f;
    param.max_new_tokens = 60;
    param.max_context_len = 1024;  // ✅ 프롬프트 최대 토큰 수 설정

    if (rkllm_init(&llmHandle, &param, callback) != 0) {
        fprintf(stderr, "[Fatal] Failed to initialize LLM.\n");
        return -1;
    }

    vector<string> pre_input = {
        "What is the recommended engine oil for this vehicle?",
        "How do I check the tire pressure?",
        "What are the safety precautions for this vehicle?",
        "How do I use the parking brake?",
        "What is the fuel capacity of this vehicle?"
    };

    printf("\n********** RAG Vehicle Assistant **********\n");
    for (size_t i = 0; i < pre_input.size(); ++i)
        printf("[%zu] %s\n", i, pre_input[i].c_str());
    printf("*******************************************\n");

    while (true) {
        string input_str;
        printf("\nEnter your question (or 'exit'): ");
        getline(cin, input_str);
        if (input_str == "exit") break;

        if (isdigit(input_str[0])) {
            int idx = stoi(input_str);
            if (idx >= 0 && idx < (int)pre_input.size()) input_str = pre_input[idx];
        }

        response_start_time = chrono::high_resolution_clock::now();
        string ragResponse = callRAGAPI(input_str);

        if (ragResponse.empty()) {
            printf("[Error] No response from RAG server.\n");
            continue;
        }

        Json::Value root;
        Json::Reader reader;
        if (!reader.parse(ragResponse, root) || !root.isMember("prompt") || !root["prompt"].isString()) {
            printf("[Error] Invalid RAG JSON.\n");
            continue;
        }

        rag_search_time = root.get("rag_search_time", 0.0).asDouble();
        prompt_gen_time = root.get("prompt_gen_time", 0.0).asDouble();
        bool is_manual_based = root.get("is_manual_based", false).asBool();
        g_prompt_str = root["prompt"].asString();

        // ✅ 긴 프롬프트 자르기 (문자 기준)
        if (g_prompt_str.length() > PROMPT_CHAR_LIMIT)
            g_prompt_str = g_prompt_str.substr(g_prompt_str.length() - PROMPT_CHAR_LIMIT);

        if (is_manual_based)
            printf("I'm thinking. Please wait...\n");

        static char prompt_buffer[4096];
        strncpy(prompt_buffer, g_prompt_str.c_str(), sizeof(prompt_buffer) - 1);
        prompt_buffer[sizeof(prompt_buffer) - 1] = '\0';

        RKLLMInput rkllm_input;
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = prompt_buffer;

        RKLLMInferParam infer_param;
        memset(&infer_param, 0, sizeof(infer_param));
        infer_param.mode = RKLLM_INFER_GENERATE;

        llm_start_time = chrono::high_resolution_clock::now();
        rkllm_run(llmHandle, &rkllm_input, &infer_param, nullptr);
    }

    rkllm_destroy(llmHandle);
    curl_global_cleanup();
    return 0;
}
