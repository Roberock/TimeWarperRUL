
from examples.example_demo_1_univariate import run_univariate_example_1
from examples.example_demo_2_bivariate_weibull import run_bivariate_example_2
from examples.example_demo_3_mixture import run_main_mixture_example_3
from examples.example_demo_4_using_lifelines import run_lifeline_demo_4

if __name__ == '__main__':
    # Press the green button in the gutter to run the script.
    run_univariate_example_1()
    run_bivariate_example_2()
    run_main_mixture_example_3(CASE=1)
    run_main_mixture_example_3(CASE=2)
    run_lifeline_demo_4()