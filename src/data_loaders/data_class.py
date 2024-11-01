import pandas as pd
import torch
import random

class CSVDataLoader():
    """
        Data Loader for the JEPA -- generates the necessary masked parts.
        ---
            - Load the data from the CSV
            - Normalize the data
            - Divide into Train/Test/Val
    """
    def __init__(self,
                 path_data,
                 batch_size=32,
                 series_split_size=100,
                 patch_size=10,
                 mask_ratio=0.15):

        input_variables = "OT"
        timestamp_col = "date"
        validation_fraction = 0.05
        test_fraction = 0.3

        # Load and preprocess data
        df = pd.read_csv(
            path_data,
            parse_dates=[timestamp_col],
            low_memory=False,
            sep=",",
        )

        # Normalize float columns
        fcols = df.select_dtypes("float").columns
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
        df_mean = df[fcols].mean(0)
        df_std = df[fcols].std(0)
        df[fcols] = (df[fcols] - df_mean) / df_std

        # Convert integer columns to numeric
        icols = df.select_dtypes("integer").columns
        df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

        # Sort by timestamp
        df.sort_values(by=[timestamp_col], inplace=True)

        # Split into train, validation, and test sets
        val_len = int(len(df) * validation_fraction)
        test_len = int(len(df) * test_fraction)
        train_len = len(df) - val_len - test_len
        df = torch.tensor(df[input_variables].values).float()
        train_df, val_df, test_df = torch.split(df, [train_len, val_len, test_len])

        # Store data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Parameters for patching and masking
        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.time_series_list = train_df

    def __getitem__(self, idx):
        # Split the original time series into smaller time series chunks
        ts = self.time_series_list
        num_splits = len(ts) // self.series_split_size
        split_series = [ts[i*self.series_split_size:(i+1)*self.series_split_size] for i in range(num_splits)]

        # Select the series based on the index (assuming batched access)
        selected_series = split_series[idx % len(split_series)]

        # Now divide the selected smaller time series into patches
        num_patches = len(selected_series) // self.patch_size
        patches = [selected_series[i*self.patch_size:(i+1)*self.patch_size] for i in range(num_patches)]

        # Convert patches to tensor
        patches_tensor = torch.stack(patches)

        # Create the mask for the patches
        num_masked_patches = int(num_patches * self.mask_ratio)
        mask_indices = random.sample(range(num_patches), num_masked_patches)
        non_mask_indices = [i for i in range(num_patches) if i not in mask_indices]

        # Separate masked and non-masked patches
        masked_patches = torch.stack([patches_tensor[i] for i in mask_indices])
        non_masked_patches = torch.stack([patches_tensor[i] for i in non_mask_indices])

        mask_indices = torch.tensor(mask_indices)
        non_mask_indices = torch.tensor(non_mask_indices)

        return patches_tensor, mask_indices, non_mask_indices

    def __len__(self):
        # Number of smaller time series created from the full series
        return len(self.time_series_list) // self.series_split_size


class EvaluationDataLoader():
    """
        Similar to the previous class of JEPA but rather used for evaluation in
        the downstream tasks
        ---
            - Load the data from the CSV.
            - Normalize the data
            - Divide into Train/Test/Val
    """
    def __init__(self, path_data, patch_size=32, context_size=10):
        self.patch_size = patch_size
        self.context_size = context_size

        input_variables = "OT"
        timestamp_col = "date"
        validation_fraction = 0.05
        test_fraction = 0.3

        df = pd.read_csv(path_data, parse_dates=[timestamp_col],
                                                    low_memory=False, sep=",")

        fcols = df.select_dtypes("float").columns
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")

        df_mean = df[fcols].mean(0)
        df_std = df[fcols].std(0)

        df[fcols] = (df[fcols] - df_mean) / df_std


        icols = df.select_dtypes("integer").columns
        df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

        df.sort_values(by=[timestamp_col], inplace=True)

        # Split into train, validation, and test sets
        val_len = int(len(df) * validation_fraction)
        test_len = int(len(df) * test_fraction)
        train_len = len(df) - val_len - test_len
        df = torch.tensor(df[input_variables].values).float()
        train_df, val_df, test_df = torch.split(df, [train_len, val_len, test_len])

        # Store data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.series = self.train_df

        # Split the entire time series into patches
        self.patches_tensor = self.split_into_patches(self.series, self.patch_size)

    def split_into_patches(self, series, patch_size):
        num_patches = len(series) // patch_size
        patches = [series[i * patch_size:(i + 1) * patch_size] for i in range(num_patches)]
        return torch.stack(patches)  # Shape will be (num_patches, patch_size)

    def __len__(self):
        # Number of available samples based on the context size
        return len(self.patches_tensor) - self.context_size

    def __getitem__(self, idx):
        # Here we ensure that each time we return a context window of 10 patches
        if idx + self.context_size + 1 > len(self.patches_tensor):
            raise IndexError("Index out of range for context window")

        # Get context patches (previous 10) and the target patch (next one)
        context_patches = self.patches_tensor[idx:idx + self.context_size]
        target_patch = self.patches_tensor[idx + self.context_size]

        return context_patches, target_patch
