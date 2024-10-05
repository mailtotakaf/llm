import os
import multiprocessing


def print_cors():
    # 論理コア数を取得
    logical_cores = os.cpu_count()

    # 物理コア数を取得
    physical_cores = multiprocessing.cpu_count()

    print(f"Logical Cores: {logical_cores}")
    print(f"Physical Cores: {physical_cores}")


# 並列で実行するタスク（例: 単純な数値計算）
def compute_square(x):
    return x * x


# 並列に処理するデータ
numbers = [1, 2, 3, 4, 5]

# 並列に計算を実行
if __name__ == "__main__":

    print_cors()

    with multiprocessing.Pool() as pool:
        results = pool.map(compute_square, numbers)

    print(results)  # [1, 4, 9, 16, 25]

