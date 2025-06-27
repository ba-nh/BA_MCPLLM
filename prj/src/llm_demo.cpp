#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <curl/curl.h>
#include <json/json.h>
#include <chrono>

using namespace std;
LLMHandle llmHandle = nullptr;

string my_path = "/home/odroid/Desktop/BAMCP/";
string my_model = "TinyLlama-1.1B-Chat-v1.0.rkllm";
string full_path = my_path + my_model;

// RAG 서버 설정
const string RAG_SERVER_URL = "http://localhost:5000/rag";

// libcurl 콜백 함수
size_t WriteCallback(void* contents, size_t size, size_t nmemb, string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// RAG API 호출 함수
string callRAGAPI(const string& query) {
    CURL* curl = curl_easy_init();
    string response;
    
    if (curl) {
        Json::Value requestData;
        requestData["query"] = query;
        requestData["k"] = 5;
        requestData["alpha"] = 0.5;

        Json::FastWriter writer;
        string jsonData = writer.write(requestData);

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, RAG_SERVER_URL.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
            response = "";
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    return response;
}

void exit_handler(int signal) {
    if (llmHandle != nullptr) {
        cout << "program is about to exit" << endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
    }
    exit(signal);
}

void callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_FINISH) {
        printf("\n");
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\\run error\n");
    } else if (state == RKLLM_RUN_GET_LAST_HIDDEN_LAYER) {
        if (result->last_hidden_layer.embd_size != 0 && result->last_hidden_layer.num_tokens != 0) {
            int data_size = result->last_hidden_layer.embd_size * result->last_hidden_layer.num_tokens * sizeof(float);
            std::ofstream outFile("last_hidden_layer.bin", std::ios::binary);
            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(result->last_hidden_layer.hidden_states), data_size);
                outFile.close();
                std::cout << "Data saved to last_hidden_layer.bin successfully!" << std::endl;
            } else {
                std::cerr << "Failed to open the file for writing!" << std::endl;
            }
        }
    } else if (state == RKLLM_RUN_NORMAL) {
        printf("%s", result->text);
    }
}

int main (void) {
    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");

    curl_global_init(CURL_GLOBAL_ALL);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = full_path.c_str();

    // Sampling settings
    param.top_k = 40;
    param.top_p = 0.9;
    param.temperature = 0.3;
    param.repeat_penalty = 1.2;
    param.frequency_penalty = 0.1;
    param.presence_penalty = 0.1;

    param.max_new_tokens = 200;
    param.max_context_len = 2048;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }

    vector<string> pre_input = {
        "What is the recommended engine oil for this vehicle?",
        "How do I check the tire pressure?",
        "What are the safety precautions for this vehicle?",
        "How do I use the parking brake?",
        "What is the fuel capacity of this vehicle?"
    };

    cout << "\n********************** RAG-Enhanced Vehicle Manual Assistant ********************\n";
    for (int i = 0; i < (int)pre_input.size(); i++) {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "**********************************************************************************\n";

    RKLLMInput rkllm_input;
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;

    while (true) {
        std::string input_str;
        printf("\nEnter your question (or 'exit' to quit): ");
        std::getline(std::cin, input_str);
        if (input_str == "exit") break;

        for (int i = 0; i < (int)pre_input.size(); i++) {
            if (input_str == to_string(i)) {
                input_str = pre_input[i];
                cout << "Selected: " << input_str << endl;
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        string ragResponse = callRAGAPI(input_str);
        if (ragResponse.empty()) {
            printf("Error: Could not connect to RAG server. Please make sure the server is running.\n");
            continue;
        }

        // RAG 응답에서 prompt 추출
        Json::Value root;
        Json::Reader reader;
        if (!reader.parse(ragResponse, root)) {
            cerr << "Failed to parse RAG response JSON" << endl;
            continue;
        }

        if (!root.isMember("prompt")) {
            cerr << "No prompt in RAG response." << endl;
            continue;
        }

        // 매뉴얼 기반 답변인지 확인
        bool is_manual_based = false;
        if (root.isMember("is_manual_based")) {
            is_manual_based = root["is_manual_based"].asBool();
        }

        // 매뉴얼 기반 답변일 때만 메시지 출력
        if (is_manual_based) {
            printf("I'm thinking. Please wait a moment.\n");
        }

        std::string text = root["prompt"].asString();

        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = (char *)text.c_str();

        printf("Assistant: ");
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("\n[Total answer time: %.3f seconds]\n", elapsed.count());
    }

    rkllm_destroy(llmHandle);
    curl_global_cleanup();
    return 0;
}
