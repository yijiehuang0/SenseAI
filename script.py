import subprocess
import os
pruning = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
sparsity = [.00001, .0001, .001, .1, 1]
file = open("results.txt", "a")
file.close()

for i in range(len(sparsity)):
    arg1 = sparsity[i]
    file = open("results.txt", "a")
    file.write("\n\n\n")
    file.write("Starting training with sparsity: {}\n".format(arg1))
    file.close()
    train = subprocess.Popen(['python', 'main.py', '-sr', '--epochs', '160', '--s', str(arg1)])
    train.wait()
    for j in range(len(pruning)):
        arg2 = pruning[j]
        try:
            subprocess.check_output(['python', 'prune.py', '--model', 'model_best.pth.tar', '--save', 'pruned4.pth.tar',
                                            '--percent', str(arg2)])
            print("pruned accuatly")
        except subprocess.CalledProcessError as e:
            print("failed")
            file = open("results.txt", "a")
            file.write("Failed to prune and retrain with pruning rate: {}\n\n\n".format(arg2))
            file.close()
            continue
        file = open("results.txt", "a")
        file.write("Pruning with rate: {}".format(arg2))
        file.close()
        train = subprocess.Popen(['python', 'main.py', '--refine', 'pruned4.pth.tar', '--epochs', str(40)])
        train.wait()
 
        os.remove("pruned4.pth.tar")
        os.remove("checkpoint.pth.tar")
        file = open("results.txt", "a")
        file.write("\n")
        file.close()
        print("refined")
    os.remove("model_best.pth.tar")
    os.remove("model_best_refine.pth.tar")