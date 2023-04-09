def get_wandb_project_table(project_name, entity='Awni00', attr_cols=('group', 'name'), config_cols='all', summary_cols='all'):
    import wandb
    import pandas as pd

    api = wandb.Api()

    runs = api.runs(entity + "/" + project_name)

    if summary_cols == 'all':
        summary_cols = set().union(*tuple(run.summary.keys() for run in runs))

    if config_cols == 'all':
        config_cols = set().union(*tuple(run.config.keys() for run in runs))

    all_cols = list(attr_cols) + list(summary_cols) + list(config_cols)
    if len(all_cols) > len(set(all_cols)):
        raise ValueError("There is overlap in the `config_cols`, `attr_cols`, and `summary_cols`")

    data = {key: [] for key in all_cols}

    for run in runs:
        for summary_col in summary_cols:
            data[summary_col].append(run.summary.get(summary_col, None))

        for config_col in config_cols:
            data[config_col].append(run.config.get(config_col, None))

        for attr_col in attr_cols:
            data[attr_col].append(getattr(run, attr_col, None))

    runs_df = pd.DataFrame(data)

    return runs_df