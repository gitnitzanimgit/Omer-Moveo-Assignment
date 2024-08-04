
# Mobile Communications Moveo Project

The Mobile Communications Moveo Project analyzes different machine learning clustering algorithms to identify various subjects of patent claims within the mobile communication patent sector.

## Installation

To install and run this project, follow these steps:

1.Download and Install Selenium WebDriver:

Visit the Selenium WebDriver download page,
Download the appropriate WebDriver for your browser (e.g., ChromeDriver for Google Chrome).

Follow the instructions on the website to install the WebDriver.

2.Ensure you have Jupyter Notebook installed:

    pip install notebook

## Usage/Examples

To run this project, follow these steps:

1.Set the path of your Chrome Binary in the variable:


    'chrome_options.binary_location' 

in the jupyter notebook.

2.Place the 'models' folder inside the working directory.

3.install dependencies based on requirements.txt

3.Open Jupyter Notebooks and run the projects.






## Notes

1.In order to check Task 1 (perform the web scraping manually and not use the 'claims.csv' file) make sure that:

    check_Task_1 = True
State right after download is:

    check_Task_1 = False


1.Notice the pyLDAvis.html File - It is an important graph of the chosen model when num_of_clusters = 5.


2.I haven't been able to run the models when the sample size was full. So I used 3/8's of it. If you want to run the Jupyter Notebook with the full sample size, simple change the name of 'full_claims' to 'claims' and make sure its in the working directory. Make sure to move the original 'claims' out of the working directory. That way you can rerun the Jupyter Notebook analysis with a sample size of 803 instead of 303 which will improve performance.