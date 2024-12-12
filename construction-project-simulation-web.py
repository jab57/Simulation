import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

class MaterialType(Enum):
    LUMBER = auto()
    CONCRETE = auto()
    STEEL = auto()

@dataclass
class Material:
    name: MaterialType
    current_quantity: float
    max_capacity: float
    reorder_threshold: float
    unit_cost: float
    procurement_delay: Tuple[float, float]
    consumption_rate: Tuple[float, float]

@dataclass
class Resource:
    name: str
    capacity: int
    hourly_cost: float

import numpy as np
import simpy
import random

class ConstructionProject:
    def __init__(
        self, 
        env: simpy.Environment, 
        initial_budget: float, 
        activities_duration: Dict[str, Tuple[float, float]],  # New parameter
        materials: List[Material],
        resources: List[Resource]
    ):
        self.env = env
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        
        self.activities_duration = activities_duration
        self.materials = {mat.name: mat for mat in materials}
        self.resources = {res.name: simpy.Resource(env, capacity=res.capacity) for res in resources}
        
        self.project_duration = 0
        self.material_shortages = {}
        self.budget_changes = []  # Track budget changes
        
        self.process = env.process(self.run_project())
    
    def _calculate_activity_cost(self, activity_name: str):
        """Randomly calculate the cost of an activity based on resource and material usage."""
        # Define random variability in cost for each activity (for simplicity, using a normal distribution)
        base_cost = 1000  # Base cost for each activity (can be refined)
        cost_variation = np.random.normal(loc=0, scale=200)  # Normal distribution of cost changes
        activity_cost = base_cost + cost_variation
        
        # Ensure non-negative cost
        return max(activity_cost, 0)
    
    def run_project(self):
        """Main simulation process for the construction project."""
        start_time = self.env.now
        
        # Perform construction activities sequentially
        yield from self.site_preparation()
        yield from self.foundation_work()
        yield from self.structural_framing()
        yield from self.roofing()
        yield from self.interior_work()
        yield from self.finishing()
        
        # Update the final project duration
        self.project_duration = self.env.now - start_time
    
    def site_preparation(self):
        """Simulate site preparation phase."""
        min_duration, max_duration = self.activities_duration.get('site_preparation', (50, 100))
        duration = np.random.uniform(min_duration, max_duration)
        
        # Calculate and deduct activity cost
        activity_cost = self._calculate_activity_cost('site_preparation')
        self.current_budget -= activity_cost
        self.budget_changes.append(activity_cost)  # Record budget change for this activity
        
        yield self.env.timeout(duration)

    def foundation_work(self):
        """Simulate foundation work phase."""
        min_duration, max_duration = self.activities_duration.get('foundation_work', (100, 200))
        duration = np.random.uniform(min_duration, max_duration)
        
        # Calculate and deduct activity cost
        activity_cost = self._calculate_activity_cost('foundation_work')
        self.current_budget -= activity_cost
        self.budget_changes.append(activity_cost)
        
        yield self.env.timeout(duration)

    def structural_framing(self):
        """Simulate structural framing phase."""
        min_duration, max_duration = self.activities_duration.get('structural_framing', (150, 250))
        duration = np.random.uniform(min_duration, max_duration)
        
        # Calculate and deduct activity cost
        activity_cost = self._calculate_activity_cost('structural_framing')
        self.current_budget -= activity_cost
        self.budget_changes.append(activity_cost)
        
        yield self.env.timeout(duration)

    def roofing(self):
        """Simulate roofing phase."""
        min_duration, max_duration = self.activities_duration.get('roofing', (80, 120))
        duration = np.random.uniform(min_duration, max_duration)
        
        # Calculate and deduct activity cost
        activity_cost = self._calculate_activity_cost('roofing')
        self.current_budget -= activity_cost
        self.budget_changes.append(activity_cost)
        
        yield self.env.timeout(duration)

    def interior_work(self):
        """Simulate interior work phase."""
        min_duration, max_duration = self.activities_duration.get('interior_work', (200, 300))
        duration = np.random.uniform(min_duration, max_duration)
        
        # Calculate and deduct activity cost
        activity_cost = self._calculate_activity_cost('interior_work')
        self.current_budget -= activity_cost
        self.budget_changes.append(activity_cost)
        
        yield self.env.timeout(duration)

    def finishing(self):
        """Simulate project finishing phase."""
        min_duration, max_duration = self.activities_duration.get('finishing', (100, 150))
        duration = np.random.uniform(min_duration, max_duration)
        
        # Calculate and deduct activity cost
        activity_cost = self._calculate_activity_cost('finishing')
        self.current_budget -= activity_cost
        self.budget_changes.append(activity_cost)
        
        yield self.env.timeout(duration)

def run_simulations(
    num_iterations: int,
    initial_budget: float,
    materials_config: List[Dict],
    resources_config: List[Dict]
):
    simulation_results = []
    
    default_activities_duration = {
        'site_preparation': (50, 100),
        'foundation_work': (100, 200),
        'structural_framing': (150, 250),
        'roofing': (80, 120),
        'interior_work': (200, 300),
        'finishing': (100, 150)
    }
    
    for _ in range(num_iterations):
        env = simpy.Environment()
        
        materials = [
            Material(
                name=MaterialType[mat['name']], 
                current_quantity=mat['current_quantity'], 
                max_capacity=mat['max_capacity'],
                reorder_threshold=mat['reorder_threshold'],
                unit_cost=mat['unit_cost'],
                procurement_delay=mat['procurement_delay'],
                consumption_rate=mat['consumption_rate']
            ) for mat in materials_config
        ]
        
        resources = [
            Resource(
                name=res['name'], 
                capacity=res['capacity'], 
                hourly_cost=res['hourly_cost']
            ) for res in resources_config
        ]
        
        project = ConstructionProject(
            env, 
            initial_budget=initial_budget, 
            activities_duration=default_activities_duration,
            materials=materials,
            resources=resources
        )
        
        env.run()
        
        simulation_results.append({
            'final_budget': project.current_budget,
            'total_project_duration': project.project_duration,
            'budget_changes': project.budget_changes,
            'material_shortages': project.material_shortages
        })
    
    return pd.DataFrame(simulation_results)



def create_material_input_section():
    """Create Streamlit inputs for material configuration."""
    st.subheader("Material Configuration")
    
    # Default material types
    material_types = [
        MaterialType.LUMBER.name, 
        MaterialType.CONCRETE.name, 
        MaterialType.STEEL.name
    ]
    
    materials_config = []
    for material_type in material_types:
        with st.expander(f"{material_type} Configuration"):
            current_quantity = st.number_input(
                f"{material_type} Current Quantity", 
                min_value=0.0, 
                value=300.0, 
                key=f"{material_type}_current_qty"
            )
            max_capacity = st.number_input(
                f"{material_type} Max Capacity", 
                min_value=0.0, 
                value=600.0, 
                key=f"{material_type}_max_capacity"
            )
            reorder_threshold = st.number_input(
                f"{material_type} Reorder Threshold", 
                min_value=0.0, 
                value=100.0, 
                key=f"{material_type}_reorder_threshold"
            )
            unit_cost = st.number_input(
                f"{material_type} Unit Cost", 
                min_value=0.0, 
                value=50.0, 
                key=f"{material_type}_unit_cost"
            )
            
            st.write("Procurement Delay (Min, Max hours)")
            proc_delay_min = st.number_input(
                f"{material_type} Procurement Delay Min", 
                min_value=0.0, 
                value=24.0, 
                key=f"{material_type}_proc_delay_min"
            )
            proc_delay_max = st.number_input(
                f"{material_type} Procurement Delay Max", 
                min_value=0.0, 
                value=48.0, 
                key=f"{material_type}_proc_delay_max"
            )
            
            st.write("Consumption Rate (Min, Max)")
            consumption_min = st.number_input(
                f"{material_type} Consumption Rate Min", 
                min_value=0.0, 
                value=10.0, 
                key=f"{material_type}_consumption_min"
            )
            consumption_max = st.number_input(
                f"{material_type} Consumption Rate Max", 
                min_value=0.0, 
                value=30.0, 
                key=f"{material_type}_consumption_max"
            )
            
            materials_config.append({
                'name': material_type,
                'current_quantity': current_quantity,
                'max_capacity': max_capacity,
                'reorder_threshold': reorder_threshold,
                'unit_cost': unit_cost,
                'procurement_delay': (proc_delay_min, proc_delay_max),
                'consumption_rate': (consumption_min, consumption_max)
            })
    
    return materials_config

def create_resource_input_section():
    """Create Streamlit inputs for resource configuration."""
    st.subheader("Resource Configuration")
    
    resources_config = []
    resource_types = ['Workers', 'Heavy Equipment']
    
    for resource_type in resource_types:
        with st.expander(f"{resource_type} Configuration"):
            capacity = st.number_input(
                f"{resource_type} Capacity", 
                min_value=1, 
                value=20 if resource_type == 'Workers' else 5, 
                key=f"{resource_type}_capacity"
            )
            hourly_cost = st.number_input(
                f"{resource_type} Hourly Cost", 
                min_value=0.0, 
                value=50.0 if resource_type == 'Workers' else 200.0, 
                key=f"{resource_type}_hourly_cost"
            )
            
            resources_config.append({
                'name': resource_type.lower().replace(' ', '_'),
                'capacity': capacity,
                'hourly_cost': hourly_cost
            })
    
    return resources_config




def visualize_simulation_results(results_df: pd.DataFrame):
    """Create visualizations for simulation analysis using Streamlit."""
    
    # Check if 'final_budget' exists and is numeric
    if 'final_budget' not in results_df.columns:
        print("Error: 'final_budget' column is missing from the data.")
        return
    
    # Ensure the data is numeric and drop any invalid rows
    results_df['final_budget'] = pd.to_numeric(results_df['final_budget'], errors='coerce')
    results_df.dropna(subset=['final_budget'], inplace=True)

    # Ensure final_budget is a 1D array-like (pandas series)
    if not isinstance(results_df['final_budget'], pd.Series):
        print("Error: 'final_budget' is not a valid pandas Series.")
        return

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Flatten the axes array for easier 1D indexing
    axes = axes.flatten()

    # Budget distribution
    try:
        sns.histplot(data=results_df, x='final_budget', kde=True, ax=axes[0], color='skyblue', bins=30, stat='density')
        axes[0].set_title('Final Budget Distribution')
        axes[0].set_xlabel('Budget ($)')
        axes[0].set_ylabel('Density')
        axes[0].grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        print(f"Error processing budget data: {e}")

    # Project Duration Distribution
    try:
        # Ensure that 'total_project_duration' is numeric and handle non-numeric entries
        results_df['total_project_duration'] = pd.to_numeric(results_df['total_project_duration'], errors='coerce')
        duration_data = results_df['total_project_duration'].dropna()
        sns.histplot(duration_data, kde=True, ax=axes[1], color='lightgreen')
        axes[1].set_title('Project Duration Distribution')
        axes[1].set_xlabel('Duration (hours)')
        axes[1].set_ylabel('Frequency')
    except Exception as e:
        print(f"Error processing duration data: {e}")

    # Budget Changes
    try:
        # Ensure that 'budget_changes' contains lists or numerical values
        budget_changes = results_df['budget_changes'].apply(lambda x: sum(x) if isinstance(x, list) else x)
        sns.histplot(budget_changes, kde=True, ax=axes[2], color='salmon')
        axes[2].set_title('Total Budget Changes')
        axes[2].set_xlabel('Budget Change ($)')
        axes[2].set_ylabel('Frequency')
    except Exception as e:
        print(f"Error processing budget changes: {e}")

    # Summary statistics
    try:
        summary_stats = results_df.describe()
        axes[3].axis('off')  # Turn off the 4th subplot
        axes[3].text(0.1, 0.5, f"Summary Statistics:\n{summary_stats.to_string()}", fontsize=10, verticalalignment='center')
    except Exception as e:
        print(f"Error processing summary statistics: {e}")

    # Adjust layout and show the plot
    plt.tight_layout()
    st.pyplot(fig)






# Example usage
# results_df = pd.DataFrame({
#     'final_budget': np.random.normal(100000, 10000, 100),
#     'total_project_duration': [np.random.normal(200, 50, 1) for _ in range(100)],
#     'budget_changes': [np.random.normal(0, 100, 10).tolist() for _ in range(100)]
# })
# visualize_simulation_results(results_df)




def main():
    st.title("Construction Project Simulation")
    
    # Sidebar for global simulation parameters
    st.sidebar.header("Simulation Parameters")
    
    # Number of simulation iterations
    num_iterations = st.sidebar.number_input(
        "Number of Simulation Iterations", 
        min_value=10, 
        max_value=1000, 
        value=100, 
        key="num_iterations"
    )
    
    # Initial budget
    initial_budget = st.sidebar.number_input(
        "Initial Project Budget", 
        min_value=10000.0, 
        value=500000.0, 
        key="initial_budget"
    )
    
    st.sidebar.header("Construction Activities Duration")
    activities = [
        'site_preparation', 'foundation_work', 'structural_framing', 
        'roofing', 'interior_work', 'finishing'
    ]
    
    activities_duration = {}
    for activity in activities:
        with st.sidebar.expander(f"{activity.replace('_', ' ').title()} Duration"):
            min_duration = st.number_input(
                f"{activity.replace('_', ' ').title()} Min Duration (hours)", 
                min_value=0.0, 
                value=50.0, 
                key=f"{activity}_min_duration"
            )
            max_duration = st.number_input(
                f"{activity.replace('_', ' ').title()} Max Duration (hours)", 
                min_value=0.0, 
                value=100.0, 
                key=f"{activity}_max_duration"
            )
            activities_duration[activity] = (min_duration, max_duration)
    
    # Material configuration section
    materials_config = create_material_input_section()
    
    # Resource configuration section
    resources_config = create_resource_input_section()
    
    # Run simulation button
    if st.button("Run Simulation"):
        with st.spinner('Running Simulation...'):
            # Run simulations with user-defined parameters
            simulation_results = run_simulations(
                num_iterations=num_iterations,
                initial_budget=initial_budget,
                materials_config=materials_config,
                resources_config=resources_config
            )
            
            # Display results
            st.subheader("Simulation Results")
            
            # Display summary statistics
            st.dataframe(simulation_results.describe())
            
            # Create and display visualizations
            fig = visualize_simulation_results(simulation_results)
            st.pyplot(fig)
            
            # Optional: Download results
            csv = simulation_results.to_csv(index=False)
            st.download_button(
                label="Download Simulation Results (CSV)",
                data=csv,
                file_name='construction_project_simulation_results.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()