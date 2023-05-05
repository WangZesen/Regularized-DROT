#ifndef utils_READER_HPP_
#define utils_READER_HPP_

#include <fstream>
#include <vector>

namespace utils {
    template <typename T>
    std::vector<T> load(const std::string &filename,
            const int n_rows, const int n_cols) {

        std::ifstream ifs(filename, std::ifstream::binary);
        std::vector<T> values(std::size_t(n_rows) * n_cols);
        std::vector<T> out(std::size_t(n_rows) * n_cols);
        ifs.read(reinterpret_cast<char *>(&values[0]),
            std::size_t(n_rows) * n_cols * sizeof(T));
        ifs.close();

        for(int col=0; col < n_cols; col++) {
            for(int row=0; row < n_rows; row++) {
                out[col * n_rows + row] = values[row * n_cols + col];
            }
        }
        return out;
    }

    template <typename T>
    void calc_mean_std(std::vector<T> data, T *_mean, T *_std) {
        T mean = 0, std = 0;
        int offset = min((int) (data.size() - 1), 1);
        if (data.size() == 0) {
            *_mean = 0;
            *_std = 0;
            return;
        }
        for (int i = offset; i < data.size(); i++) {
            mean += data[i];
        }
        mean = mean / (T) (data.size() - offset);
        *_mean = mean;
        for (int i = offset; i < data.size(); i++) {
            std += (data[i] - mean) * (data[i] - mean);
        }
        std = sqrt(std / (T) (data.size() - offset));
        *_std = std;
    }

    template <typename T>
    void save_trans(const T* x, std::string filename,
            const int n_rows, const int n_cols) {

        FILE *stream;
        stream = fopen(filename.c_str(), "w");

        for(int row=0; row < n_rows; row++) {
            for(int col=0; col < n_cols; col++) {
                fprintf(stream, "%.10f ", x[col * n_rows + row]);
            }
            fprintf(stream, "\n");
        }
        fclose(stream);
    }

};
#endif
