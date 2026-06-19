import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("artifacts/plots", exist_ok=True)

# 1. Histogram (Salary Distribution)
def salary_histogram(data):
    plt.figure(figsize=(8, 5))
    plt.hist(data['salary_in_usd'], bins=20)
    plt.title("Salary Distribution")
    plt.xlabel("Salary in USD")
    plt.ylabel("Frequency")
    plt.savefig("artifacts/plots/salary_distribution.png")
    plt.close()


# 2. Box Plot (Outlier Detection)
def salary_boxplot(data):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data['salary_in_usd'])
    plt.title("Box Plot of Salary")
    plt.savefig("artifacts/plots/salary_boxplot.png")
    plt.close()


# 3. Count Plot (Experience Levels)
def experience_countplot(data):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='experience_level', data=data)
    plt.title("Experience Level Count")
    plt.savefig("artifacts/plots/experience_countplot.png")
    plt.close()



# 4. Bar Plot (Average Salary by Company Size)
def company_size_barplot(data):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x='company_size',
        y='salary_in_usd',
        data=data
    )
    plt.title("Average Salary by Company Size")
    plt.savefig("artifacts/plots/company_size_barplot.png")
    plt.close()


# 5. Scatter Plot (Remote Ratio vs Salary)
def remote_salary_scatter(data):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        data['remote_ratio'],
        data['salary_in_usd']
    )
    plt.title("Remote Ratio vs Salary")
    plt.xlabel("Remote Ratio")
    plt.ylabel("Salary in USD")
    plt.savefig("artifacts/plots/remote_salary_scatter.png")
    plt.close()


# 6. Violin Plot (Salary by Employment Type)
def salary_violinplot(data):
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x='employment_type',
        y='salary_in_usd',
        data=data
    )
    plt.title("Salary Distribution by Employment Type")
    plt.savefig("artifacts/plots/salary_violinplot.png")
    plt.close()


# 7. Line Plot (Average Salary by Work Year)
def salary_year_lineplot(data):
    year_salary = data.groupby(
        'work_year'
    )['salary_in_usd'].mean()

    plt.figure(figsize=(8, 5))
    plt.plot(
        year_salary.index,
        year_salary.values,
        marker='o'
    )
    plt.title("Average Salary by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Salary")
    plt.savefig("artifacts/plots/salary_year_lineplot.png")
    plt.close()


# 8. Pie Chart (Company Size Distribution)
def company_size_piechart(data):
    size_counts = data['company_size'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.pie(
        size_counts,
        labels=size_counts.index,
        autopct='%1.1f%%'
    )
    plt.title("Company Size Distribution")
    plt.savefig("artifacts/plots/company_size_piechart.png")
    plt.close()



# 9. Heatmap (Correlation Matrix)
def correlation_heatmap(data):
    corr = data[
        ['work_year',
         'salary',
         'salary_in_usd',
         'remote_ratio']
    ].corr()

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm'
    )
    plt.title("Correlation Heatmap")
    plt.savefig("artifacts/plots/correlation_heatmap.png")
    plt.close()



# 10. KDE Plot (Salary Density)
def salary_kdeplot(data):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data['salary_in_usd'],
        fill=True
    )
    plt.title("Salary Density Curve")
    plt.savefig("artifacts/plots/salary_kdeplot.png")
    plt.close()