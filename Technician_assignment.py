import streamlit as st
import sys
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import io
import requests # Added for the sample file URL
import zipfile # Added for creating zip files

# --- Class Definitions ---
class Technician():
    def __init__(self, name, cap, depot):
        self.name = name
        self.cap = cap
        self.depot = depot

    def __str__(self):
        return f"Technician: {self.name}\n Capacity: {self.cap}\n Depot: {self.depot}"

class Job():
    def __init__(self, name, priority, duration, coveredBy):
        self.name = name
        self.priority = priority
        self.duration = duration
        self.coveredBy = coveredBy

    def __str__(self):
        about = f"Job: {self.name}\n Priority: {self.priority}\n Duration: {self.duration}\n Covered by: "
        about += ", ".join([t.name for t in self.coveredBy])
        return about

class Customer():
    def __init__(self, name, loc, job, tStart, tEnd, tDue):
        self.name = name
        self.loc = loc
        self.job = job
        self.tStart = tStart
        self.tEnd = tEnd
        self.tDue = tDue

    def __str__(self):
        coveredBy = ", ".join([t.name for t in self.job.coveredBy])
        return f"Customer: {self.name}\n Location: {self.loc}\n Job: {self.job.name}\n Priority: {self.job.priority}\n Duration: {self.job.duration}\n Covered by: {coveredBy}\n Start time: {self.tStart}\n End time: {self.tEnd}\n Due time: {self.tDue}"

# --- Data Loading and Processing Functions ---
@st.cache_data
def load_data(excel_file):
    """
    Loads data from the uploaded Excel file into Technician, Job, and Customer objects.
    """
    xls = pd.ExcelFile(excel_file)

    # Read Technician data
    df_tech = pd.read_excel(xls, sheet_name='Technicians')
    df_tech = df_tech.rename(columns={df_tech.columns[0]: "name", df_tech.columns[1]: "cap", df_tech.columns[2]: "depot"})
    df_tech_filtered = df_tech.drop(df_tech.columns[3:], axis=1).drop(df_tech.index[[0,1]])
    technicians = [Technician(*row) for row in df_tech_filtered.itertuples(index=False, name=None)]

    # Read job data
    jobs = []
    for j in range(3, len(df_tech.columns)):
        coveredBy = [t for i, t in enumerate(technicians) if df_tech.iloc[2+i,j]==1]
        thisJob = Job(df_tech.iloc[2:,j].name, df_tech.iloc[0,j], df_tech.iloc[1,j], coveredBy)
        jobs.append(thisJob)

    # Read location data
    df_locations = pd.read_excel(xls, sheet_name='Locations', index_col=0)
    locations = df_locations.index
    dist = {(l, l): 0 for l in locations}
    for i, l1 in enumerate(locations):
        for j, l2 in enumerate(locations):
            if i < j:
                dist[l1, l2] = df_locations.iloc[i, j]
                dist[l2, l1] = dist[l1, l2]

    # Read customer data
    df_customers = pd.read_excel(xls, sheet_name='Customers')
    customers = []
    for i, c in enumerate(df_customers.iloc[:, 0]):
        job_name = df_customers.iloc[i, 2]
        matching_job = next((job for job in jobs if job.name == job_name), None)
        if matching_job is not None:
            this_customer = Customer(c, df_customers.iloc[i, 1], matching_job, *df_customers.iloc[i, 3:])
            customers.append(this_customer)
    
    return technicians, customers, dist, jobs, df_tech_filtered, df_locations, df_customers

def get_latest_times(customers, technician, dist):
    """
    Calculates the latest possible start times for each customer in a given route.
    """
    latest = dict()
    if not customers:  # Handle case where customers list is empty
        return latest

    d = dist[customers[-1].loc, technician.depot]
    prev_latest = min(customers[-1].tEnd, 600 - d - customers[-1].job.duration)
    latest[customers[-1].loc] = prev_latest
    for i in range(len(customers) - 2, -1, -1):
        d = dist[customers[i].loc, customers[i + 1].loc]
        latest_end = min(prev_latest - d - customers[i].job.duration, customers[i].tEnd)
        latest[customers[i].loc] = latest_end
        prev_latest = latest_end
    return latest


def solve_trs0(technicians, customers, dist):
    """
    Solves the Technician Routing and Scheduling problem using Gurobi.
    Returns assignment details, routes, and utilization.
    """
    K = [k.name for k in technicians]
    C = [j.name for j in customers]
    J = [j.loc for j in customers]
    L = list(set([l[0] for l in dist.keys()]))
    D = list(set([t.depot for t in technicians]))
    cap = {k.name: k.cap for k in technicians}
    loc = {j.name: j.loc for j in customers}
    depot = {k.name: k.depot for k in technicians}
    canCover = {j.name: [k.name for k in j.job.coveredBy] for j in customers}
    dur = {j.name: j.job.duration for j in customers}
    tStart = {j.name: j.tStart for j in customers}
    tEnd = {j.name: j.tEnd for j in customers}
    priority = {j.name: j.job.priority for j in customers}

    m = gp.Model("trs0")
    m.setParam('OutputFlag', 0) # Suppress Gurobi output in Streamlit

    x = m.addVars(C, K, vtype=GRB.BINARY, name="x")
    u = m.addVars(K, vtype=GRB.BINARY, name="u")
    y = m.addVars(L, L, K, vtype=GRB.BINARY, name="y")
    t = m.addVars(L, ub=600, name="t")
    g = m.addVars(C, vtype=GRB.BINARY, name="g")

    for k in technicians:
        for d in D:
            if k.depot != d:
                for i in L:
                    y[i, d, k.name].ub = 0
                    y[d, i, k.name].ub = 0

    m.addConstrs((gp.quicksum(x[j, k] for k in canCover[j]) + g[j] == 1 for j in C), name="assignToJob")
    m.addConstrs((x.sum(j, '*') <= 1 for j in C), name="assignOne")

    capLHS = {k: gp.quicksum(dur[j] * x[j, k] for j in C) + \
                  gp.quicksum(dist[i, j] * y[i, j, k] for i in L for j in L) for k in K}
    m.addConstrs((capLHS[k] <= cap[k] * u[k] for k in K), name="techCapacity")

    m.addConstrs((y.sum('*', loc[j], k) == x[j, k] for k in K for j in C), name="techTour1")
    m.addConstrs((y.sum(loc[j], '*', k) == x[j, k] for k in K for j in C), name="techTour2")

    m.addConstrs((gp.quicksum(y[j, depot[k], k] for j in J) == u[k] for k in K), name="sameDepot1")
    m.addConstrs((gp.quicksum(y[depot[k], j, k] for j in J) == u[k] for k in K), name="sameDepot2")

    M_val_cust = {(i, j): 600 + dur[i] + dist[loc[i], loc[j]] for i in C for j in C}
    m.addConstrs((t[loc[j]] >= t[loc[i]] + dur[i] + dist[loc[i], loc[j]] \
                  - M_val_cust[i, j] * (1 - gp.quicksum(y[loc[i], loc[j], k] for k in K)) \
                  for i in C for j in C), name="tempoCustomer")

    M_val_depot = {(i, j): 600 + dist[i, loc[j]] for i in D for j in C}
    m.addConstrs((t[loc[j]] >= t[i] + dist[i, loc[j]] \
                  - M_val_depot[i, j] * (1 - y.sum(i, loc[j], '*')) for i in D for j in C), \
                  name="tempoDepot")

    m.addConstrs((t[loc[j]] >= tStart[j] for j in C), name="timeWinA")
    m.addConstrs((t[loc[j]] <= tEnd[j] for j in C), name="timeWinB")

    M_obj = 6100
    m.setObjective(gp.quicksum(M_obj * priority[j] * g[j] for j in C) + gp.quicksum(0.01 * M_obj * t[loc_key] for loc_key in t.keys() if loc_key in J),
                   GRB.MINIMIZE)

    m.optimize()

    assignment_results = []
    route_results = []
    utilization_results = []
    routes_list = []
    orders_list = []

    status = m.Status
    if status in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED]:
        assignment_results.append("Model is either infeasible or unbounded.")
    elif status != GRB.OPTIMAL:
        assignment_results.append("Optimization terminated with status {}".format(status))
    else:
        # Assignments
        for j in customers:
            if g[j.name].X > 0.5:
                jobStr = "Nobody assigned to {} ({}) in {}".format(j.name,j.job.name,j.loc)
            else:
                for k in K:
                    if x[j.name, k].X > 0.5:
                        jobStr = f"**{k}** assigned to **{j.name}** ({j.job.name}) in {j.loc}. Start at t={t[j.loc].X:.2f}."
            assignment_results.append(jobStr)

        # Technicians routes and preparing data for excel output
        routes_cols = ['Route ID', 'Technician Name', 'Origin Location', 'Total Travel Time', 'Total Processing Time',
                       'Total Time', 'Earliest Start Time', 'Latest Start Time', 'Earliest End Time',
                       'Latest End Time', 'Num Jobs']
        orders_cols = ['Route ID', 'Stop Number', 'Customer Name', 'Technician Name', 'Location Name', 'Job type',
                       'Processing Time', 'Customer Time Window Start', 'Customer Time Window End',
                       'Earliest Start', 'Latest Start', 'Earliest End', 'Latest End', ]
        route_id = 0

        for k_obj in technicians:
            k_name = k_obj.name
            if u[k_name].X > 0.5:
                route_results.append(f"**{k_name}'s route:**")
                current_location = k_obj.depot
                route_str = k_obj.depot
                
                # Collect customers in the route
                current_customers_in_route = []
                temp_current_location = k_obj.depot # Use a temp variable to build the customer list
                while True:
                    customer_found_in_segment = False
                    for j in customers:
                        if y[temp_current_location, j.loc, k_name].X > 0.5:
                            current_customers_in_route.append(j)
                            temp_current_location = j.loc
                            customer_found_in_segment = True
                            break
                    if not customer_found_in_segment:
                        break # No more customers in this segment

                total_travel_time, total_processing_time = 0, 0
                current_location = k_obj.depot
                stop_number = 0
                
                # Build route string and orders_list
                for j in current_customers_in_route:
                    total_travel_time += dist[current_location, j.loc]
                    total_processing_time += j.job.duration
                    stop_number += 1
                    route_str += (f" -> {j.loc} (dist={dist[current_location, j.loc]}, t={t[j.loc].X:.2f},"
                                  f" proc={j.job.duration}, a={tStart[j.name]}, b={tEnd[j.name]})")
                    current_location = j.loc
                    
                    # Prepare data for orders_list
                    latest_times = get_latest_times(current_customers_in_route, k_obj, dist)
                    
                    orders_list.append([route_id + 1, stop_number, j.name, k_name, j.loc, j.job.name,
                                        j.job.duration, tStart[j.name], tEnd[j.name],
                                        t[j.loc].X, latest_times.get(j.loc, 'N/A'), # Using .get for safety
                                        t[j.loc].X + j.job.duration, latest_times.get(j.loc, 'N/A') + j.job.duration if latest_times.get(j.loc) else 'N/A'])
                
                # Add return to depot to route string and travel time
                if current_customers_in_route:
                    travel_back_to_depot = dist[current_location, k_obj.depot]
                    total_travel_time += travel_back_to_depot
                    route_str += f" -> {k_obj.depot} (dist={travel_back_to_depot})"
                else: # Technician used but no customers assigned, e.g., only travel
                    travel_back_to_depot = 0
                    
                route_results.append(route_str)
                route_id += 1

                # Prepare data for routes_list
                earliest_start_route, latest_start_route, earliest_end_route, latest_end_route = 0,0,0,0
                if current_customers_in_route:
                    earliest_start_route = t[current_customers_in_route[0].loc].X - dist[k_obj.depot, current_customers_in_route[0].loc]
                    
                    # Recalculate latest times for the specific route for accurate route-level values
                    latest_times_for_route = get_latest_times(current_customers_in_route, k_obj, dist)
                    latest_start_route = latest_times_for_route[current_customers_in_route[0].loc] - dist[k_obj.depot, current_customers_in_route[0].loc]

                    earliest_end_route = t[current_customers_in_route[-1].loc].X + current_customers_in_route[-1].job.duration + dist[current_customers_in_route[-1].loc, k_obj.depot]
                    latest_end_route = latest_times_for_route[current_customers_in_route[-1].loc] + current_customers_in_route[-1].job.duration + dist[current_customers_in_route[-1].loc, k_obj.depot]
                
                routes_list.append([route_id, k_obj.name, k_obj.depot, total_travel_time, total_processing_time,
                                     earliest_end_route - earliest_start_route, earliest_start_route,
                                     latest_start_route, earliest_end_route, latest_end_route, len(current_customers_in_route)])
            else:
                route_results.append(f"**{k_name}** is not used")

        # Utilization
        totUsed = sum(capLHS[k].getValue() for k in K)
        totCap = sum(cap[k] for k in K)
        totUtil = totUsed / totCap if totCap > 0 else 0
        utilization_results.append(f"Total technician utilization is **{totUtil:.2%}** ({totUsed:.2f}/{totCap:.2f})")
        for k in K:
            used = capLHS[k].getValue()
            total = cap[k]
            util = used / total if total > 0 else 0
            utilization_results.append(f"**{k}**'s utilization is **{util:.2%}** ({used:.2f}/{total:.2f})")

    # Dispose Gurobi model
    m.dispose()
    gp.disposeDefaultEnv()
    
    routes_df = pd.DataFrame.from_records(routes_list, columns=routes_cols)
    orders_df = pd.DataFrame.from_records(orders_list, columns=orders_cols)

    return assignment_results, route_results, utilization_results, routes_df, orders_df

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Technician Assignment Optimizer")

st.markdown("""
Upload your Excel file to optimize technician assignments. 
You can view a sample Excel file to understand the required format.
""")

# Sample file redirection
sample_excel_view_url = "https://docs.google.com/spreadsheets/d/16pxWguPkBtw4f4lZiFkqplxprqgIIkvj/edit?usp=sharing"
st.markdown(f"**<a href='{sample_excel_view_url}' target='_blank'>Click here to view the Sample Excel File</a>**", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    if st.button("Run Optimization"):
        with st.spinner("Optimizing... This might take a moment."):
            try:
                technicians, customers, dist, jobs, df_tech_filtered, df_locations, df_customers = load_data(uploaded_file)
                assignment_results, route_results, utilization_results, routes_df, orders_df = solve_trs0(technicians, customers, dist)

                st.subheader("Optimization Results")

                st.markdown("### Customer Assignments")
                for result in assignment_results:
                    st.markdown(result)

                st.markdown("### Technician Routes")
                for result in route_results:
                    st.markdown(result)

                st.markdown("### Utilization")
                for result in utilization_results:
                    st.markdown(result)

                # Combined Download Button
                st.subheader("Download Results")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr("routes.csv", routes_df.to_csv(index=False).encode('utf-8'))
                    zip_file.writestr("orders.csv", orders_df.to_csv(index=False).encode('utf-8'))
                zip_buffer.seek(0)

                st.download_button(
                    label="Download All Results (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="technician_assignment_results.zip",
                    mime="application/zip"
                )

            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")
                st.warning("Please ensure your Excel file is formatted correctly according to the sample file.")

else:
    st.info("Please upload an Excel file to start the optimization.")
