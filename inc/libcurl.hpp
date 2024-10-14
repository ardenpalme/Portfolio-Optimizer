#ifndef __LIBCURL_HPP__
#define __LIBCURL_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <curl/curl.h>

class CURLError : public std::exception {
    int status;
public:
    CURLError(int _status) : status{_status} {}

    const char *what() const noexcept {
        std::string status_str;
        if(status == -1) {
            status_str = "CURL initialization error";
        } else if(status >= CURL_LAST) {
                status_str = "Undefined CURL error";
        }else{
            status_str = std::string(curl_easy_strerror((CURLcode)status));
        }
        return status_str.c_str();
    }
};

class CURLInitError : public std::exception {
public:
    const char *what() const noexcept {
        return "CURL Initialization Error";
    }
};

class URL {
    CURL *curl;
    std::string response;

    static size_t raw_callback(char *data, size_t size, size_t nmemb, void *clientp) {
        size_t payload_size = size * nmemb;
        static_cast<URL*>(clientp)->data_callback(data, payload_size);

        return payload_size;
    }

    void data_callback(char *data, size_t size) {
        response.reserve(size);
        for(int i=0; i<size; i++) {
            response += data[i];
        }
    }

public:

    URL(std::string url) {
        curl = curl_easy_init();
        if(curl) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, raw_callback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
        }else{
            throw CURLError(-1);
        }
    }

    ~URL() {
        curl_easy_cleanup(curl);
    }

    std::shared_ptr<std::string> get_data() {
        CURLcode status;
        response.clear();
        if((status = curl_easy_perform(curl)) != CURLE_OK) {
            throw CURLError(status);
        }

        // CURL callbacks have processed all chunks
        return std::make_shared<std::string>(response);
    }

};

#endif /* __LIBCURL_HPP__ */