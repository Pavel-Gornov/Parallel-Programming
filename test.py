import subprocess
import numpy as np
import platform

def numpy_result(matrix_path: str) -> float:
    with open(matrix_path, mode="r") as f:
        data = f.readlines()[1:]
    
    matrix = []
    for line in data:
        matrix.append(list(map(float, line.split())))
    return np.linalg.det(matrix)

def cpp_result(executable_path: str, matrix_path: str) -> tuple[float, float]:
    result = subprocess.run(
        [executable_path, "data.txt"],
        capture_output=True,
        text=True,  # Decode output as text
        check=True
    )
    output = result.stdout
    dets = output.split("\n")[-1]
    return tuple(map(float, dets.split()))

def main(executable_path: str, matrix_path: str) -> None:
    det_np = numpy_result(matrix_path)
    det_1, det_2 = cpp_result(executable_path, matrix_path)
    print(det_np, det_1, det_2)
    if np.isclose(det_np, det_1) and np.isclose(det_np, det_2):
        print("Значения подсчитаны верно!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    if platform.system() == 'Windows':
        default_exec_path = "build\Debug\Parallel.exe"
    elif platform.system() == 'Linux':
        default_exec_path = "build\Parallel"

    parser.add_argument("-e", "--executable", type=str, help="Путь до исполняемого файла", default=default_exec_path)
    parser.add_argument("-d", "--data", type=str, help="Путь к файлу с матрицей", default="data.txt")

    args = parser.parse_args()

    main(args.executable, args.data)