import subprocess
import numpy as np
import platform
import random
import io

def cpp_result(executable_path: str, matrix: list[list[float]]):
    """Запускает написанную программу, и считывает вывод, возвращает результат вычисления"""
    n = len(matrix)
    input_string = io.StringIO()
    input_string.write(f"{n} {n}\n")
    for i in range(n):
        input_string.write(" ".join(map(str, matrix[i])) + "\n")

    result = subprocess.run(
        [executable_path], 
        input=input_string.getvalue(),
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout.split("result:")
    print(output[0], end="") # Результирующая матрица не выводится
    time = output[0].split("time elapsed: ")[-1]
    time = float(time.strip("ms\n"))

    o = output[1].split("\n")
    m = []
    for i in o:
        l = list(map(float, i.split()))
        if l:
            m.append(l)

    return m, time

def main(executable_path: str) -> None:
    times = []
    for n in (200, 400, 800, 1200, 1600, 2000):
        matrix = [[random.uniform(-100, 100) for _ in range(n)] for _ in range(n)]
        res_np = np.matmul(matrix, matrix)
        
        res, time = cpp_result(executable_path, matrix)
        times.append((n, time))
        #print(res, res_np)
        if (np.allclose(res_np, res)):
            print("[test]: Значения подсчитаны верно!")
        else:
            print(f"[test]: Результаты не совпадают: {res_np} {res}")
        print()
    print(times)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    if platform.system() == 'Windows':
        default_exec_path = "build\Debug\Parallel.exe"
    elif platform.system() == 'Linux':
        default_exec_path = "./build/Parallel"

    parser.add_argument("-e", "--executable", type=str, help="Путь до исполняемого файла", default=default_exec_path)

    args = parser.parse_args()

    main(args.executable)