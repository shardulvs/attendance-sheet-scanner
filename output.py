def print_2d_array(array):
    if not array or not array[0]:
        print("Empty or invalid array")
        return
    column_widths = [max(len(str(item)) for item in col) for col in zip(*array)]
    header = " | ".join([f"Col {i}".ljust(column_widths[i]) for i in range(len(array[0]))])
    print(header)
    print("-" * len(header))
    for row in array:
        print(" | ".join([str(item).ljust(column_widths[i]) for i, item in enumerate(row)]))