import pandas as pd
import matplotlib.pyplot as plt

from sagemaker.analytics import TrainingJobAnalytics

def get_dfs_from_hpt(summaries, metrics):
    '''
    Helper function to get a list of dataframes from a HyperparameterTuningJobAnalytics summary
    Parameters:
    -----------
    summaries: {}
        Output of training_job_summaries() of the HyperparameterTuningJobAnalytics object
        
    metrics: [str]
        A list of names of the metrics (e.g., "test_auc")
        
    Returns:
    --------
    res: [(str, pd.DataFrame)]
        A list of dataframe where str is the jobname and pd.DataFrame is the correponding data.
    '''
    res = []
    for summary in summaries:
        job_name = summary["TrainingJobName"]
        job_df = pd.DataFrame()

        for m in metrics:
            df = TrainingJobAnalytics(job_name, [m]).dataframe()
            df.rename(columns={'value':'{}'.format(m)}, inplace=True)
            if len(df) == 0:
                continue
            del df["metric_name"]
            timestamp = df["timestamp"]
            del df["timestamp"]
            # Ensure that there are at least 500 epochs
            job_df = pd.concat([job_df, df], 1)
            job_df["timestamp"] = timestamp/60
        res.append((job_name, job_df))
    return res

def plot_df_list(df_list, metric_name, y_label, min_final_value):
    '''
    Helper function to plot the performance of a list of jobs
    Parameters:
    -----------
    df_list: [(str, pd.DataFrame)]
        A list of dataframe where str is the jobname and pd.DataFrame is the correponding data.
        
    metric_name: str
        Name of the metric used
        
    y_label: str
        y_label for the plot
        
    min_final_value: float
        Only plots training jobs that reached the specified value
    '''
    my_dpi = 108
    fig = plt.figure(figsize=(1000/my_dpi, 800/my_dpi), dpi=my_dpi)

    linewidth = 3
    font_size = 24
    
    x = "Epoch"
    
    for job_name, job_df in df_list:
        if metric_name not in job_df.columns:
            continue
        final_value = job_df[metric_name].values[-1]
        if final_value > min_final_value:
            plt.plot(job_df[x], job_df[metric_name], label=job_name, linewidth=linewidth)
    plt.xlabel(x, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.grid(color="0.9", linestyle='-', linewidth=3)
    
    plt.tight_layout()
    
def get_best_training_job(df_list, metric, maximize_or_minimize):
    '''
    Helper function to get the best training job.
    '''
    assert maximize_or_minimize in ["maximize", "minimize"], "maximize_or_minimize must be either 'maximize' or 'minimize'"
    if maximize_or_minimize == "maximize":
        best_value = 0
    else:
        best_value = 1e5
        
    best_job = None
        
    for job_name, job_df in df_list:
        if metric not in job_df.columns:
            continue
        final_value = job_df[metric].values[-1]
        if maximize_or_minimize == "maximize":
            if final_value > best_value:
                best_job = job_name, job_df
        else:
            if final_value < best_value:
                best_job = job_name, job_df
    return best_job


    
