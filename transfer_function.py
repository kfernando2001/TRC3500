# Example USAGE
# python transfer_function.py real_world_demo.csv 1 2 --x_column 0
# where 1 refers to column 1 of CSV, 2 refers to column 2 of CSV
# --x_column 0 refers to column 0 of CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def analyze_and_plot(df, x_column, y_columns):
    Vref = 3.3      # Reference voltage
    ADC_max = 4095  # STM32 12-bit ADC

    # Define x-values
    if x_column is not None:
        x = df[x_column].values
        # x_moisture = ((x / np.max(x))) * 100    # Convert ml to moisture percentage
    else:
        x = np.arange(len(df))                  # Use index if x_column is not specified
        # x_moisture = np.arange(len(df))
    
    # Extract & stage y-values for use
    y_values = df[y_columns].values
    y_voltage = (y_values / ADC_max) * Vref                                                                     # Convert ADC to voltage
    x_moisture = (1 - (np.abs(y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values)))) * 100      # Convert ADC to moisture percentage
    y_mean = np.mean(y_values, axis=1)
    y_voltage_mean = np.mean(y_voltage, axis=1)

    # OPTION1 : Compute ideal line using terminal points method
    # m_ideal = (y_mean[-1] - y_mean[0]) / (x[-1] - x[0])
    # b_ideal = y_mean[0] - m_ideal * x[0]
    # y_ideal = m_ideal * x + b_ideal

    # OPTION2 : Compute ideal line with best fit line
    m_ideal, b_ideal = np.polyfit(x, y_mean, 1)
    m_ideal_voltage, b_ideal_voltage = np.polyfit(x, y_voltage_mean, 1)
    y_ideal = m_ideal * x + b_ideal
    y_ideal_voltage = m_ideal_voltage * x + b_ideal_voltage

    # Find maximum deviation from ideal line
    deviations = np.abs(y_values - y_ideal[:, np.newaxis])
    deviations_voltage = np.abs(y_voltage - y_ideal_voltage[:, np.newaxis])
    max_deviation = np.max(deviations)
    max_deviation_voltage = np.max(deviations_voltage)
    # terminal_deviations = np.abs(y_mean - y_ideal)                              # Use if terminal points method
    # terminal_deviations_voltage = np.abs(y_voltage_mean - y_ideal_voltage)      # Use if terminal points method
    # terminal_max_deviation = np.max(terminal_deviations)                        # Use if terminal points method
    # terminal_max_deviation_voltage = np.max(terminal_deviations_voltage)        # Use if terminal points method

    # Compute upper and lower bound lines
    y_upper = y_ideal + max_deviation
    y_upper_voltage = y_ideal_voltage + max_deviation_voltage
    y_lower = y_ideal - max_deviation
    y_lower_voltage = y_ideal_voltage - max_deviation_voltage

    # Compute R² value
    ss_total = np.sum((y_mean - np.mean(y_mean)) ** 2)
    ss_residual = np.sum((y_mean - y_ideal) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # Compute required sensor performance metrics
    full_range = np.max(y_values) - np.min(y_values)
    span = compute_span(x)
    repeatability = compute_repeatability(y_values, full_range)         # Repeatability error: Lower is better
    inaccuracy = (np.max(deviations) / full_range) * 100                # Accuracy error: Lower is better
    nonlinearity = (max_deviation / np.max(y_ideal)) * 100              # Nonlinearity: Lower is better
    deviation_percentage = (max_deviation / np.max(y_ideal)) * 100      # Deviation percentage: Lower is better

    # Print sensor performance metrics
    print(f'Span: {span}')
    print(f'Equation of Ideal Line (ADC vs Water Volume): y = {m_ideal:.4f}x + {b_ideal:.4f}')
    print(f'Equation of Ideal Line (Voltage vs Water Volume): y = {m_ideal_voltage:.4f}x + {b_ideal_voltage:.4f}')
    print(f'R² Value: {r2:.4f}')
    # print(f'Positive/Negative Deviation: {max_deviation:.4f}')
    print(f'Positive/Negative Deviation Percentage: {deviation_percentage:.4f}%')
    print(f'Repeatability: {repeatability:.2f}%')
    print(f'Inaccuracy: {inaccuracy:.2f}%')
    print(f'Nonlinearity: {nonlinearity:.2f}%')

    # Plot Voltage vs Water volume (ml)
    plt.figure(1, figsize=(10, 5))
    for i, y_col in enumerate(y_columns):
        plt.scatter(x, (df[y_col] / ADC_max * Vref), label=f'Data {i+1}', alpha=0.6)
    plt.scatter(x, y_voltage_mean, label='Mean Data', color='black', marker='x', s=100)
    plt.plot(x, y_ideal_voltage, label='Ideal Line', color='red', linewidth=2)
    plt.plot(x, y_upper_voltage, label='Positive Deviation', color='blue', linestyle='dashed')
    plt.plot(x, y_lower_voltage, label='Negative Deviation', color='blue', linestyle='dashed')
    
    equation_text_voltage = f'Span = {span:.2f}\n$y = {m_ideal_voltage:.4f}x + {b_ideal_voltage:.4f}$\nPositive/Negative Deviation = {deviation_percentage:.2f}%\n$R^2 = {r2:.2f}$\nRepeatability = {repeatability:.2f}%\nInaccuracy = {inaccuracy:.2f}%\nNonlinearity = {nonlinearity:.2f}%'
    plt.text(
    min(x) + 0.05 * (max(x) - min(x)),
    min(y_voltage_mean) + 0.05 * (max(y_voltage_mean) - min(y_voltage_mean)),  # Adjusted text position
    equation_text_voltage,
    fontsize=12,
    color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )
    
    plt.xlabel('Water Volume (ml)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Voltage vs Water Volume (ml)')
    plt.legend()
    plt.grid()

    # Plot ADC vs Water volume (ml)
    plt.figure(2, figsize=(10, 5))
    for i, y_col in enumerate(y_columns): 
        plt.scatter(x, df[y_col], label=f'Data {i+1}', alpha=0.6)
    plt.scatter(x, y_mean, label='Mean Data', color='black', marker='x', s=100)
    plt.plot(x, y_ideal, label='Ideal Line', color='red', linewidth=2)
    plt.plot(x, y_upper, label='Positive Deviation', color='blue', linestyle='dashed')
    plt.plot(x, y_lower, label='Negative Deviation', color='blue', linestyle='dashed')

    equation_text_adc = f'Span = {span:.2f}\nADC range = {full_range}\n$y = {m_ideal:.4f}x + {b_ideal:.4f}$\nPositive/Negative Deviation = {deviation_percentage:.2f}%\n$R^2 = {r2:.2f}$\nRepeatability = {repeatability:.2f}%\nInaccuracy = {inaccuracy:.2f}%\nNonlinearity = {nonlinearity:.2f}%'
    plt.text(
    min(x) + 0.05 * (max(x) - min(x)),
    min(y_mean) + 0.05 * (max(y_mean) - min(y_mean)),  # Adjusted text position
    equation_text_adc,
    fontsize=12,
    color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )
    
    plt.xlabel('Water Volume (ml)')
    plt.ylabel('ADC Value')
    plt.title(f'ADC Value vs Water Volume (ml)')
    plt.legend()
    plt.grid()

    # Plot Moisture level vs Water volume (ml)
    x_moisture_mean = np.mean(x_moisture, axis=1)
    m_ideal_moisture, b_ideal_moisture = np.polyfit(x, x_moisture_mean, 1)
    y_ideal_moisture = m_ideal_moisture * x + b_ideal_moisture

    plt.figure(3, figsize=(10, 5))
    plt.scatter(x, x_moisture_mean, label='Mean Data', color='black', marker='x', s=100)
    plt.plot(x, y_ideal_moisture, label='Ideal Line', color='red', linewidth=2)
    
    print(f'Equation of Ideal Line (Moisture Percentage vs Water Volume): y = {m_ideal_moisture:.4f}x {b_ideal_moisture:.4f}')

    test_water_volume = 22 # [DEMO] Water volume (ml) added to soil

    full_water_volume = (100 - b_ideal_moisture) / m_ideal_moisture
    print(f'At 100% moisture level, water volume: {full_water_volume:.4f} ml')
    print(f'[FOR DEMO] With {test_water_volume} ml of water, moisture level: {m_ideal_moisture * test_water_volume + b_ideal_moisture:.4f}%')

    equation_text_moisture = f'Span = {span:.2f}\n$y = {m_ideal_moisture:.4f}x {b_ideal_moisture:.4f}$\nPositive/Negative Deviation = {deviation_percentage:.2f}%\n$R^2 = {r2:.2f}$\nRepeatability = {repeatability:.2f}%\nInaccuracy = {inaccuracy:.2f}%\nNonlinearity = {nonlinearity:.2f}%'
    plt.text(
    min(x) + 0.05 * (max(x) - min(x)),
    max(x_moisture_mean) - 0.25 * (max(x_moisture_mean) - min(x_moisture_mean)),  # Adjusted text position
    equation_text_moisture,
    fontsize=12,
    color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    plt.xlabel('Water Volume (ml)')
    plt.ylabel('Moisture Percentage (%)')
    plt.title(f'Moisture Percentage (%) vs Water Volume (ml)')
    plt.legend()
    plt.grid()
    plt.show()


def compute_repeatability(y_values, full_range):
    y_max = np.max(y_values, axis=1)
    y_min = np.min(y_values, axis=1)
    repeatability = np.mean((y_max - y_min) / full_range) * 100
    return repeatability


def compute_span(x_values):
    x_values = np.array(x_values, dtype=np.float64)
    span = abs(np.max(x_values) - np.min(x_values))
    if span == 0:
        span = 1
    return span


if __name__ == "__main__":
    # Perform analysis and plotting
    parser = argparse.ArgumentParser(description="Perform linear regression on selected columns from CSV.")
    parser.add_argument("csv_file", help="Path to the CSV file containing data.")
    parser.add_argument("y_columns", type=int, nargs='+', help="Indices of the columns to use as Y values (0-based). Can provide multiple.")
    parser.add_argument("--x_column", type=int, default=None, help="(Optional) Index of the column to use as X values (0-based). If not specified, the index is used.")
    args = parser.parse_args()
    
    # Read CSV
    df = pd.read_csv(args.csv_file, header=None)

    # Validate column indices
    for y_col in args.y_columns:
        if y_col < 0 or y_col >= len(df.columns):
            print(f"Error: Y column index {y_col} is out of range. The CSV has {len(df.columns)} columns.")
            exit(1)

    if args.x_column is not None and (args.x_column < 0 or args.x_column >= len(df.columns)):
        print(f"Error: X column index {args.x_column} is out of range. The CSV has {len(df.columns)} columns.")
        exit(1)

    # Assign column names
    df.columns = [f"Column_{i}" for i in range(len(df.columns))]
    y_column_names = [f"Column_{i}" for i in args.y_columns]
    x_column_name = f"Column_{args.x_column}" if args.x_column is not None else None

    # Convert selected columns to numeric
    for y_col in y_column_names:
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    if x_column_name:
        df[x_column_name] = pd.to_numeric(df[x_column_name], errors="coerce")
    
    # Drop NaN values
    df = df.dropna()

    # Compute span using x-values
    span = compute_span(df[x_column_name].values if x_column_name else df.index.values)

    analyze_and_plot(df, x_column_name, y_column_names)

