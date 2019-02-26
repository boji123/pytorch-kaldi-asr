import subprocess

def get_available_gpu_ids():
    result = subprocess.getoutput('nvidia-smi --query-gpu=index,memory.used,utilization.gpu  --format=csv')
    #print(result)
    result = result.split('\n')[1:]

    available_gpu_ids = []
    for line in result:
        arr = line.split(',')
        memoryused = arr[1].split()[0]
        gpuused = arr[2].split()[0]
        if int(memoryused) <= 20 and int(gpuused) <= 2:
            available_gpu_ids += [int(arr[0])]

    return available_gpu_ids