# Example USAGE
# python transfer_function_v2.py wk2_assignment.csv 1 --x_column 0
# where 1 refers to column 1 of CSV, --x_column 0 refers to column 0 of CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def analyze_and_plot(df, x_column, y_columns):
    Vref = 3.3  # Reference voltage
    ADC_max = 4095  # Assuming a 12-bit ADC

    # Define x-values
    if x_column is not None:
        x = df[x_column].values
        x = (1 - (x / np.max(x))) * 100  # Convert raw ADC to moisture percentage
    else:
        x = np.arange(len(df))  # Use index if x_column is not specified
    
    # Scale y-values (ADC) to voltage
    y_values = df[y_columns].values
    y_values = (y_values / ADC_max) * Vref  # Convert ADC to voltage
    
    # OPTION1 : Compute ideal line using terminal points method
    y_mean = np.mean(y_values, axis=1)
    # m_ideal = (y_mean[-1] - y_mean[0]) / (x[-1] - x[0])
    # b_ideal = y_mean[0] - m_ideal * x[0]
    # y_ideal = m_ideal * x + b_ideal

    # OPTION2 : Compute ideal line with best fit line
    m_ideal, b_ideal = np.polyfit(x, y_mean, 1)
    y_ideal = m_ideal * x + b_ideal

    # Find maximum deviation from ideal line
    deviations = np.abs(y_values - y_ideal[:, np.newaxis])
    max_deviation = np.max(deviations)
    terminal_deviations = np.abs(y_mean - y_ideal)
    terminal_max_deviation = np.max(terminal_deviations)

    # Compute upper and lower bound lines
    y_upper = y_ideal + max_deviation
    y_lower = y_ideal - max_deviation

    # Compute R² value
    ss_total = np.sum((y_mean - np.mean(y_mean)) ** 2)
    ss_residual = np.sum((y_mean - y_ideal) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # Compute repeatability, inaccuracy and nonlinearity
    # full_range = np.max(y_values)
    full_range = np.max(y_values) - np.min(y_values)
    if full_range == 0:
        full_range = 1
    print(f'Full Range: {full_range}')
    span = compute_span(x)
    repeatability = compute_repeatability(y_values, full_range) # Lower is better
    inaccuracy = (np.sum(deviations) / (len(deviations) * span)) * 100 # 
    nonlinearity = (terminal_max_deviation / np.max(y_ideal)) * 100

    # Print equation and max deviation
    print(f'Span: {span}')
    print(f'Equation of Ideal Line: y = {m_ideal:.4f}x + {b_ideal:.4f}')
    print(f'R² Value: {r2:.4f}')
    print(f'Positive/Negative Deviation: {max_deviation:.4f}')
    print(f'Repeatability: {repeatability:.2f}%')
    print(f'Inaccuracy: {inaccuracy:.2f}%')
    print(f'Nonlinearity: {nonlinearity:.2f}%')

    # Plot data
    plt.figure(figsize=(10, 5))
    # for i, y_col in enumerate(y_columns):
    #     plt.scatter(x, df[y_col], label=f'Data {i+1}', alpha=0.6)
    for i, y_col in enumerate(y_columns): 
        plt.scatter(x, (df[y_col] / 4096) * 3.3, label=f'Data {i+1}', alpha=0.6)
    plt.scatter(x, y_mean, label='Mean Data', color='black', marker='x', s=100)
    plt.plot(x, y_ideal, label='Ideal Line', color='red', linewidth=2)
    plt.plot(x, y_upper, label='Positive Deviation', color='blue', linestyle='dashed')
    plt.plot(x, y_lower, label='Negative Deviation', color='blue', linestyle='dashed')
    
    equation_text = f'Span = {span:.2f}\n$y = {m_ideal:.4f}x + {b_ideal:.4f}$\nPositive/Negative Deviation = {max_deviation:.4f}\n$R^2 = {r2:.2f}$\nRepeatability = {repeatability:.2f}%\nInaccuracy = {inaccuracy:.2f}%\nNonlinearity = {nonlinearity:.2f}%'
    plt.text(min(x) + 0.05 * (max(x) - min(x)), max(y_mean) * 0.9, equation_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.xlabel('Moisture Percentage (%)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Data Analysis: {y_columns} vs Moisture Percentage')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot original data
    # plt.figure(figsize=(10, 5))
    # for i, y_col in enumerate(y_columns):
    #     plt.scatter(x, df[y_col], label=f'Data {i+1}', alpha=0.6)
    # plt.scatter(x, y_mean, label='Mean Data', color='black', marker='x', s=100)
    # plt.plot(x, y_ideal, label='Ideal Line', color='red', linewidth=2)
    # plt.plot(x, y_upper, label='Positive Deviation', color='blue', linestyle='dashed')
    # plt.plot(x, y_lower, label='Negative Deviation', color='blue', linestyle='dashed')

    # # Display equation and max deviation on plot
    # equation_text = f'Span = {span:.2f}\n$y = {m_ideal:.4f}x + {b_ideal:.4f}$\nPositive/Negative Deviation = {max_deviation:.4f}\n$R^2 = {r2:.2f}$\nRepeatability = {repeatability:.2f}%\nInaccuracy = {inaccuracy:.2f}%\nNonlinearity = {nonlinearity:.2f}%'
    # plt.text(min(x) + 0.05 * (max(x) - min(x)), max(y_mean) * 0.9, equation_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # ADC real world test
    # x_expected = 3.5
    # y_expected = m_ideal*x_expected + b_ideal
    # print(f'Expected value for ADC = {y_expected} at {x_expected} ml')

    # Plot distances as vertical lines
    # for xi, yi, y_fit in zip(x, y_mean, y_ideal):
    #     plt.plot([xi, xi], [yi, y_fit], color='gray', linestyle='dashed', alpha=0.6)

    # plt.xlabel(x_column if x_column else 'Index')
    # plt.ylabel(y_columns)
    # plt.title(f'Data Analysis: {y_columns} vs {x_column if x_column else "Index"}')
    # plt.legend()
    # plt.grid()
    # plt.show()


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
    # print(f'Span: {span}')
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

