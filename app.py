from flask import Flask, render_template, request, session
from flask_session import Session  # Import Flask-Session
import numpy as np
import matplotlib
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "Dlathgml"  # Replace with your own secret key

# Configure server-side session
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, color="blue", label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms of slopes and intercepts
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.5, label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label="Observed Slope")
    plt.axvline(intercept, color="orange", linestyle="-", linewidth=1, label="Observed Intercept")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        (X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme, slopes, intercepts) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"], session["Y"] = X.tolist(), Y.tolist()
        session["slope"], session["intercept"] = slope, intercept
        session["slopes"], session["intercepts"] = slopes, intercepts
        session["slope_extreme"], session["intercept_extreme"] = slope_extreme, intercept_extreme
        session["N"], session["mu"], session["sigma2"] = N, mu, sigma2
        session["beta0"], session["beta1"], session["S"] = beta0, beta1, S

        return render_template("index.html", plot1=plot1, plot2=plot2, slope_extreme=slope_extreme, intercept_extreme=intercept_extreme, N=N, mu=mu, sigma2=sigma2, beta0=beta0, beta1=beta1, S=S)

    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    if not session.get("N") or not session.get("slope") or not session.get("intercept"):
        return "Please generate data first by using the 'Generate Data' button."

    N, S = int(session["N"]), int(session["S"])
    slope, intercept = float(session["slope"]), float(session["intercept"])
    slopes, intercepts = session["slopes"], session["intercepts"]
    beta0, beta1 = float(session["beta0"]), float(session["beta1"])

    parameter = request.form["parameter"]
    test_type = request.form["test_type"]

    simulated_stats = np.array(slopes if parameter == "slope" else intercepts)
    observed_stat = slope if parameter == "slope" else intercept
    hypothesized_value = beta1 if parameter == "slope" else beta0

    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))

    fun_message = "Rare event alert!" if p_value <= 0.0001 else None

    plt.figure()
    plt.hist(simulated_stats, bins=10, alpha=0.5, label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", label=f"Observed {parameter.capitalize()}")
    plt.axvline(hypothesized_value, color="blue", linestyle="-", label=f"H0: {parameter.capitalize()}={hypothesized_value}")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template("index.html", plot1="static/plot1.png", plot2="static/plot2.png", plot3=plot3_path, parameter=parameter, observed_stat=observed_stat, hypothesized_value=hypothesized_value, N=N, beta0=beta0, beta1=beta1, S=S, p_value=p_value, fun_message=fun_message)

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    if not session.get("N") or not session.get("slope") or not session.get("intercept"):
        return "Please generate data first by using the 'Generate Data' button."

    N = int(session["N"])
    mu, sigma2 = float(session["mu"]), float(session["sigma2"])
    beta0, beta1 = float(session["beta0"]), float(session["beta1"])
    S = int(session["S"])
    slopes, intercepts = session["slopes"], session["intercepts"]

    parameter = request.form["parameter"]
    confidence_level = float(request.form["confidence_level"])

    estimates = np.array(slopes if parameter == "slope" else intercepts)
    true_param = beta1 if parameter == "slope" else beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    t_value = t.ppf((1 + confidence_level / 100) / 2, len(estimates) - 1)
    ci_lower = mean_estimate - t_value * (std_estimate / np.sqrt(len(estimates)))
    ci_upper = mean_estimate + t_value * (std_estimate / np.sqrt(len(estimates)))

    includes_true = (ci_lower <= true_param <= ci_upper)

    plt.figure()
    plt.scatter(estimates, [0] * len(estimates), color="gray", alpha=0.5, label="Simulated Estimates")
    plt.errorbar(mean_estimate, 0, xerr=[[mean_estimate - ci_lower], [ci_upper - mean_estimate]], fmt='o', color="blue", label="Mean Estimate")
    plt.axvline(true_param, color="green", linestyle="--", label="True Value")
    plt.hlines(0, ci_lower, ci_upper, color="blue", linewidth=2, label=f"{confidence_level}% Confidence Interval")
    plt.xlabel(f"{parameter.capitalize()} Estimate")
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()


    return render_template("index.html", plot1="static/plot1.png", plot2="static/plot2.png", plot4=plot4_path, parameter=parameter, confidence_level=confidence_level, mean_estimate=mean_estimate, ci_lower=ci_lower, ci_upper=ci_upper, includes_true=includes_true, N=N, mu=mu, sigma2=sigma2, beta0=beta0, beta1=beta1, S=S)

if __name__ == "__main__":
    app.run(debug=True)