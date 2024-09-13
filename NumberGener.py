import numpy as np


def generate_list(mean = 5, std_dev = 1, size = 20 , lower_bound = 3, upper_bound = 10, mid_lower_bound = 4, mid_upper_bound = 6):
    while True:
        data = np.random.normal(mean, std_dev, size*3)

        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

        mid_range_count = np.sum((filtered_data >= mid_lower_bound) & (filtered_data <= mid_upper_bound))
        total_count = len(filtered_data)

        desired_mid_range_count = total_count // 2

        while mid_range_count < desired_mid_range_count:
            new_data = np.random.normal(mean, std_dev, size)
            new_filtered_data = new_data[(new_data >= lower_bound) & (new_data <= upper_bound)]
            new_mid_range_data = new_filtered_data[(new_filtered_data >= mid_lower_bound) & (new_filtered_data <= mid_upper_bound)]
            additional_data_needed = desired_mid_range_count - mid_range_count
            if len(new_mid_range_data) > additional_data_needed:
                new_mid_range_data = new_mid_range_data[:additional_data_needed]
            filtered_data = np.concatenate((filtered_data, new_mid_range_data))
            mid_range_count = np.sum((filtered_data >= mid_lower_bound) & (filtered_data <= mid_upper_bound))
            total_count = len(filtered_data)

        while mid_range_count > desired_mid_range_count:
            mid_range_indices = np.where((filtered_data >= mid_lower_bound) & (filtered_data <= mid_upper_bound))[0]
            excess_count = mid_range_count - desired_mid_range_count
            remove_indices = np.random.choice(mid_range_indices, excess_count, replace=False)
            filtered_data = np.delete(filtered_data, remove_indices)
            mid_range_count = np.sum((filtered_data >= mid_lower_bound) & (filtered_data <= mid_upper_bound))
            total_count = len(filtered_data)

        result_list = filtered_data.tolist()

        result_list = [int(x) for x in result_list][:size]

        if len(result_list) == size:
            break

    # print(result_list)
    return result_list
