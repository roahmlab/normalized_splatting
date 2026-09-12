"""Weights & Biases logging.

wandb is imported lazily so that it is only required when --wandb is passed.
"""


class WandbWriter:
    def __init__(self, project_name, run_name, tags, output_dir=None):
        import wandb
        self._wandb = wandb

        self.start_wandb = wandb.init(project=project_name,
                                      name=run_name,
                                      tags=tags,
                                      dir=output_dir)
        self.psnr_list = []

    def log(self, metrics, step=None):
        self._wandb.log(metrics)

    def log_image(self, image_data, step=None):
        img = self._wandb.Image(image_data, caption="Sample image")
        self._wandb.log({"outputs": img})

    def log_image_grid(self, image_grid, grid_caption, heading):
        img = self._wandb.Image(image_grid, caption=grid_caption)
        self._wandb.log({heading: img})

    def log_psnr_chart(self, psnr, iter):
        self.psnr_list.append(psnr)

    def plot_psnr_histogram(self):
        self._wandb.log({"PSNR_Histogram": self._wandb.Histogram(self.psnr_list)})

    def finish(self):
        self.plot_psnr_histogram()
        self._wandb.finish()
