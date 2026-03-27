import subprocess
import numpy as np
import platform
import random
import io

def cpp_result(executable_path: str, mpi_workers: int, matrix: list[list[float]]) -> float:
    """Запускает написанную программу, и считывает вывод, возвращает результат вычисления"""
    n = len(matrix)
    input_string = io.StringIO()
    input_string.write(f"{n} {n}\n")
    for i in range(n):
        input_string.write(" ".join(map(str, matrix[i])) + "\n")

    result = subprocess.run(
        # TODO: Как-нибудь обобщить это для mpirun и аргумента -np
        # А пока только так ¯\_(ツ)_/¯
        ["mpiexec", "-n", f"{mpi_workers}", executable_path], 
        input=input_string.getvalue(),
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout
    print(output, end="")
    result = output.split("\n")[-3].split(": ")[-1]
    if result == "-nan(ind)":
        return float('nan')
    return float(result)

def main(executable_path: str) -> None:
    for threads in (1, 2, 4, 8, 16):
        for n in (200, 400, 800, 1200, 1600, 2000):
            matrix = [[random.uniform(-0.1, 0.1) for _ in range(n)] for _ in range(n)]
            # print(*matrix, sep="\n")

            det_np = np.linalg.det(matrix)
            det_program = cpp_result(executable_path, threads, matrix)

            if np.isclose(det_np, det_program):
                print("[test]: Значения подсчитаны верно!")
            else:
                print(f"[test]: Результаты не совпадают: f{det_np, det_program}")
            print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    if platform.system() == 'Windows':
        default_exec_path = "build\Debug\Parallel.exe"
    elif platform.system() == 'Linux':
        default_exec_path = "build\Parallel"

    parser.add_argument("-e", "--executable", type=str, help="Путь до исполняемого файла", default=default_exec_path)

    args = parser.parse_args()

    main(args.executable)