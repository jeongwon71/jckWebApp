from flask import Flask, render_template, request, send_file, jsonify
from jck_calculation import jck_versus_num_exp
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import io
import csv

app = Flask(__name__)

global_plot_data = None

def load_data(files):
    data = {}
    for key in files:
        file = files[key]
        data[key] = np.loadtxt(io.BytesIO(file.read()), delimiter=',')
    return data

def process_data(data):
    global mu_c
    global result
    mu_c = data.get('mu_c')
    sigma_c = data.get('sigma_c')
    mu_e = data.get('mu_e')
    sigma_e = data.get('sigma_e')
    meas_sample = data.get('meas_sample')
    calc_sample = data.get('calc_sample')
    
    if mu_c is not None and mu_e is not None and sigma_c is not None and sigma_e is not None:
        result = jck_versus_num_exp(mu_c, mu_e, sigma_c, sigma_e)
    elif mu_c is not None and sigma_c is not None and meas_sample is not None:
        mu_e = np.mean(meas_sample, axis=0)
        sigma_e = np.cov(meas_sample.T)
        result = jck_versus_num_exp(mu_c, mu_e, sigma_c, sigma_e)
    elif mu_e is not None and sigma_e is not None and calc_sample is not None:
        mu_c = np.mean(calc_sample, axis=0)
        sigma_c = np.cov(calc_sample.T)
        result = jck_versus_num_exp(mu_c, mu_e, sigma_c, sigma_e)
    elif calc_sample is not None and meas_sample is not None:
        mu_c = np.mean(calc_sample, axis=0)
        sigma_c = np.cov(calc_sample.T)
        mu_e = np.mean(meas_sample, axis=0)
        sigma_e = np.cov(meas_sample.T)
        result = jck_versus_num_exp(mu_c, mu_e, sigma_c, sigma_e)
    else:
        return {"error": "Missing or invalid files"}
    
    number_of_experiments = list(range(len(mu_e) + 1))
    
    table_data = []
    for method in ['ck-ascending','ck-descending','optimal(jk)']:
        pred_k = result[method]["pred_k"]
        pred_u = result[method]["pred_u"]
        table_data.append([{"experiment": n, "pred_k": pk, "pred_u": ps} for n, pk, ps in zip(number_of_experiments, pred_k, pred_u)])
    # print(table_data)
    return {"table_data": table_data, "plot_url":"/plot"}

@app.route('/', methods=['GET', 'POST'])
def index():
    with open('bnmkinf.txt', 'r') as file:
        items = [line.rstrip() for line in file.readlines()]
    return render_template('index.html', items=items)

@app.route('/download_calc_keff', methods=['POST'])
def download_calc_keff():
    data = request.json
    indices = data.get('indices', [])
    calc_keff = np.genfromtxt('calc_keff.csv', delimiter=',')
    selected_keff_values = calc_keff[indices]
    np.savetxt('selected_calc_keff.csv', selected_keff_values, delimiter=',')
    return send_file('selected_calc_keff.csv', as_attachment=True)
    
@app.route('/download_meas_keff', methods=['POST'])
def download_meas_keff():
    data = request.json
    indices = data.get('indices', [])
    calc_keff = np.genfromtxt('meas_keff.csv', delimiter=',')
    selected_keff_values = calc_keff[indices[:-1]]
    np.savetxt('selected_meas_keff.csv', selected_keff_values, delimiter=',')
    return send_file('selected_meas_keff.csv', as_attachment=True)

@app.route('/download_calc_cov', methods=['POST'])
def download_calc_cov():
    data = request.json
    indices = data.get('indices', [])
    calc_cov = np.loadtxt('calc_cov.csv', delimiter=',')
    selected_calc_cov = calc_cov[indices][:,indices]
    np.savetxt('selected_calc_cov.csv', selected_calc_cov, delimiter=',')
    return send_file('selected_calc_cov.csv', as_attachment=True)

@app.route('/download_meas_cov', methods=['POST'])
def download_meas_cov():
    data = request.json
    indices = data.get('indices', [])
    meas_cov = np.loadtxt('meas_cov.csv', delimiter=',')
    selected_meas_cov = meas_cov[indices[:-1]][:,indices[:-1]]
    np.savetxt('selected_meas_cov.csv', selected_meas_cov, delimiter=',')
    return send_file('selected_meas_cov.csv', as_attachment=True)

@app.route('/process_sum', methods=['POST'])
def process_sum():
    files = request.files
    data = load_data(files)
    response = process_data(data)
    return jsonify(response)

@app.route('/plot')
def plot():
    global result
    if result is None:
        return jsonify({"error": "No plot data available"})

    methods = ['ck-ascending','ck-descending','optimal(jk)']

    plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])  

    ax1 = plt.subplot(gs[0, :])  
    ax1.set_title("Merged")
    lgd = [method+" order" for method in methods]
    num_exp = np.arange(len(result[methods[0]]['pred_k']))
    for i in range(len(methods)):
        ax1.errorbar(num_exp - 0.3*(i-1), result[methods[i]]['pred_k'], yerr=result[methods[i]]['pred_u'],
                    fmt='s',
                    capsize=5,
                    capthick=1,
                    markerfacecolor='red',
                    markeredgewidth=1,
                    markeredgecolor='black',
                    linestyle='-')
        ax1.legend(lgd)

    ax2 = plt.subplot(gs[1, 0])  
    ax2.errorbar(num_exp, result[methods[0]]['pred_k'], yerr=result[methods[0]]['pred_u'],
                    fmt='s',
                    capsize=5,
                    capthick=1,
                    markerfacecolor='red',
                    markeredgewidth=1,
                    markeredgecolor='black',
                    linestyle='-')
    ax2.set_title("ck-ascending sorting")

    ax3 = plt.subplot(gs[1, 1])  
    ax3.errorbar(num_exp, result[methods[1]]['pred_k'], yerr=result[methods[1]]['pred_u'],
                    fmt='s',
                    capsize=5,
                    capthick=1,
                    markerfacecolor='red',
                    markeredgewidth=1,
                    markeredgecolor='black',
                    linestyle='-')
    ax3.set_title("ck-descending sorting")

    ax4 = plt.subplot(gs[1, 2])  
    ax4.errorbar(num_exp, result[methods[2]]['pred_k'], yerr=result[methods[2]]['pred_u'],
                    fmt='s',
                    capsize=5,
                    capthick=1,
                    markerfacecolor='red',
                    markeredgewidth=1,
                    markeredgecolor='black',
                    linestyle='-')
    ax4.set_title("optimal(jk) sorting")

    plt.tight_layout()
    plt.show()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/download_graph')
def download_graph():
    plot()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png', as_attachment=True, download_name='graph.png')

@app.route('/download_data')
def download_data():
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['Number of experiments used','predicted k (ck)', 'predicted unc (ck)', 'predicted k (jk)', 'predicted unc (jk)', 'predicted k (rand)', 'predicted unc (rand)'])
    num_rows = len(result['ck']['pred_k'])

    for i in range(num_rows):
        row = [
            i,
            result['ck-ascending']['pred_k'][i], result['ck-ascending']['pred_u'][i], 
            result['ck-descending']['pred_k'][i], result['ck-descending']['pred_u'][i], 
            result['optimal(jk)']['pred_k'][i], result['optimal(jk)']['pred_u'][i]
        ]
        writer.writerow(row)

    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='result.csv')

@app.route("/download_demo_mu_c")
def download_demo_mu_c():
    path = 'demo_mu_c.csv'
    return send_file(path, as_attachment=True)

@app.route("/download_demo_mu_e")
def download_demo_mu_e():
    path = 'demo_mu_e.csv'
    return send_file(path, as_attachment=True)

@app.route("/download_demo_Sig_c")
def download_demo_Sig_c():
    path = 'demo_Sig_c.csv'
    return send_file(path, as_attachment=True)

@app.route("/download_demo_Sig_e")
def download_demo_Sig_e():
    path = 'demo_Sig_e.csv'
    return send_file(path, as_attachment=True)

@app.route("/download_demo_simul_sample")
def download_demo_simul_sample():
    path = 'demo_simul_sample.csv'
    return send_file(path, as_attachment=True)

@app.route("/download_demo_meas_sample")
def download_demo_meas_sample():
    path = 'demo_meas_sample.csv'
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
