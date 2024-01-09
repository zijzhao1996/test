def determine_years(test_year):
    """Determine training and validation years based on the test year."""
    train_years = [str(test_year - 3), str(test_year - 2)]  # Last two years before the test year
    valid_years = [str(test_year - 1)]  # Year immediately before the test year
    return train_years, valid_years