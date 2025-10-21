import argparse
import sys
from pathlib import Path

import pandas as pd


def get_csv_files(directory: Path) -> dict[str, Path]:
    """Get all CSV files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")

    csv_files = list(directory.glob("*.csv"))
    return {f.name: f for f in csv_files}


def merge_csv_files(user_dir: Path, current_dir: Path) -> None:
    """
    Merge CSV files from user_dir and current_dir into user_dir.

    The user_dir is treated as the "original directory" where merged results are saved.
    Data from current_dir is merged into the files in user_dir.

    Args:
        user_dir: Path to user-specified directory (target for merged files)
        current_dir: Path to current directory (source of new data)
    """
    # Get CSV files from both directories
    user_csvs = get_csv_files(user_dir)
    current_csvs = get_csv_files(current_dir)

    # Find common CSV files
    common_files = set(user_csvs.keys()) & set(current_csvs.keys())

    if not common_files:
        print("No matching CSV files found between the directories.")
        return

    print(f"Found {len(common_files)} matching CSV files:")
    for filename in sorted(common_files):
        print(f"  - {filename}")

    merged_count = 0

    for filename in sorted(common_files):
        try:
            print(f"\nProcessing {filename}...")

            # Read both CSV files
            user_df = pd.read_csv(user_csvs[filename])
            current_df = pd.read_csv(current_csvs[filename])

            print(
                f"  Original CSV (user dir): {len(user_df)} rows, {len(user_df.columns)} columns"
            )
            print(
                f"  New data CSV (current dir): {len(current_df)} rows, {len(current_df.columns)} columns"
            )

            # Get all unique columns from both dataframes
            all_columns = list(user_df.columns) + [
                col for col in current_df.columns if col not in user_df.columns
            ]

            # Add missing columns to user_df with empty values
            for col in current_df.columns:
                if col not in user_df.columns:
                    user_df[col] = ""
                    print(f"    Added missing column '{col}' to original CSV")

            # Add missing columns to current_df with empty values
            for col in user_df.columns:
                if col not in current_df.columns:
                    current_df[col] = ""
                    print(f"    Added missing column '{col}' to new data CSV")

            # Reorder columns to match the combined column order
            user_df = user_df[all_columns]
            current_df = current_df[all_columns]

            # Concatenate the dataframes (user data first, then current data)
            merged_df = pd.concat([user_df, current_df], ignore_index=True)

            # Save the merged file back to the user directory (overwrite original)
            output_path = user_csvs[filename]
            merged_df.to_csv(output_path, index=False)

            print(
                f"  Merged: {len(merged_df)} total rows, {len(merged_df.columns)} columns"
            )
            print(f"  Saved to: {output_path}")
            merged_count += 1

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    print(f"\nSuccessfully merged {merged_count} CSV files into the user directory.")
    print(f"Files in {user_dir} have been updated with merged data.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge CSV files from current directory into user-specified directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_results.py /path/to/target/results
  python merge_results.py ../target_results

The script will:
1. Find CSV files with matching names in both directories
2. Merge data from current directory into files in the user-specified directory
3. Handle missing columns by adding them with empty values
4. Overwrite files in the user-specified directory with merged results
        """,
    )

    parser.add_argument(
        "user_directory",
        help="Path to the target directory where merged CSV files will be saved (original files will be overwritten)",
    )

    parser.add_argument(
        "--current-dir",
        default="results",
        help="Path to the source directory containing new data to merge (default: results directory)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually merging",
    )

    args = parser.parse_args()

    try:
        # Convert to absolute paths
        user_dir = Path(args.user_directory).resolve()
        current_dir = Path(args.current_dir).resolve()

        print(f"Target directory (user): {user_dir}")
        print(f"Source directory (current): {current_dir}")

        if not user_dir.exists():
            print(f"Error: Target directory '{user_dir}' does not exist.")
            sys.exit(1)

        if not current_dir.exists():
            print(f"Error: Source directory '{current_dir}' does not exist.")
            sys.exit(1)

        if args.dry_run:
            print("\n--- DRY RUN MODE ---")
            user_csvs = get_csv_files(user_dir)
            current_csvs = get_csv_files(current_dir)
            common_files = set(user_csvs.keys()) & set(current_csvs.keys())

            if common_files:
                print(f"Would merge {len(common_files)} files into target directory:")
                for filename in sorted(common_files):
                    print(f"  - {filename}")
                print(f"\nFiles in {user_dir} would be overwritten with merged data.")
            else:
                print("No matching CSV files found.")
        else:
            merge_csv_files(user_dir, current_dir)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
