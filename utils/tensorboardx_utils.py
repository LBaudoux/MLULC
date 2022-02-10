import os
from torch.utils.tensorboard import SummaryWriter
import subprocess
import time

def tensorboard_summary_writer(config,comment):
    s = SummaryWriter(log_dir=config.summary_dir, comment=comment)
    process = subprocess.Popen(["tensorboard", "--logdir=" + config.summary_dir], cwd=os.path.abspath(os.getcwd()))
    time.sleep(10)
    return s, process