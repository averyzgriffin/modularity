import glob
import os
from os.path import join
import shutil
import sys
import subprocess
import pandas as pd


def clear_dirs(id):
    folders = ['graphviz_plots', 'networkx_graphs', 'saved_weights', 'csvs']
    for folder in folders:
        dir_path = join(folder, id).replace("\\", "/")
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename).replace("\\", "/")
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


def evolve(exp_id):
    procs = []
    for j in range(9):
        for i in range(10):
            i = i + 10
            i = i + (10*j)
            proc = subprocess.Popen([sys.executable, 'main.py', '--exp_id', exp_id, '--trial_number', f'{i}'])
            procs.append(proc)

        for proc in procs:
            proc.wait()


def post_process(exp_id):
    files = os.path.join(f"csvs/{exp_id}", f"{exp_id}*.csv")
    files = glob.glob(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df = df.sort_values('Trial #')
    df.to_csv(f"csvs/{exp_id}/{exp_id}_combined.csv", index=False)


if __name__ == "__main__":
    exp_id = "noooooooooooooo051122_SuperEvolve_MVG"
    # clear_dirs(exp_id)
    evolve(exp_id)
    post_process(exp_id)


