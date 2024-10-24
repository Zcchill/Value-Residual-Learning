import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def extract_loss(log_file_path, section):
    """
    Extract loss and grad norm from log file
    return:
    extracted_data_loss: list of loss
    l1: name of the log file
    """
    l1 = log_file_path.split('/')[-1].split('.')[0].replace("modeling_","")
    tags = ['loss', 'grad_norm']
    loss_step, extracted_data_loss = [], []
    total_step= 10000

    # extract data from log file
    with open(log_file_path, 'r') as file:
        for line in file:
            try:
                if tags[0] not in line or tags[1] not in line:
                    continue
                data = eval(line.strip())
                extracted_data_loss.append(data[tags[0]])
                loss_step.append(int(total_step * data['epoch']))
            except json.JSONDecodeError:
                continue

    # moving average for training loss
    loss_step = [x+1 for x, _ in enumerate(extracted_data_loss)]
    extracted_data_loss = moving_average(extracted_data_loss, 100)

    # extract data in section
    res_x, res_y = [], []
    for x, y in zip(loss_step, extracted_data_loss):
        if x in range(section[0], section[1]):
            res_x.append(x)
            res_y.append(y)

    if len(res_x) == 0:
        return None, None
    return [res_x, res_y], l1


def plot_loss(file_ls, label, section=(0, 10000)):
    [_, extracted_data_loss1], _ = extract_loss(file_ls[0], section)
    [_, extracted_data_loss2], _ = extract_loss(file_ls[1], section)
    delta = [b - a for a, b in zip(extracted_data_loss1, extracted_data_loss2)]
    plt.plot(delta, label=label)

if __name__ == '__main__':

    file_ls_all = {
        "Exp_Name": [[
            ['baseline.log','analog/method1.log'],
            ['baseline.log','analog/method2.log'],
            ],[
            'method1',
            'method2',
            ]],
    }
    plt.figure()
    sns.set_style("darkgrid")
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Relative Training Loss', fontsize=14)
    exp_name = "Exp_Name"
    section = (0, 9950)
    for i,_ in enumerate(file_ls_all[exp_name][0]):
        plot_loss(file_ls_all[exp_name][0][i], file_ls_all[exp_name][1][i], section=section)
    plt.plot([0 for _ in range(section[1]-section[0])], color='#FF7F7F', linestyle='--')
    plt.legend(fontsize=12)
    plt.grid(True)
    # plt.ylim(-0.1, 0.05)
    plt.savefig(f"figure_v2/{exp_name}.pdf", bbox_inches='tight', dpi=300)
        