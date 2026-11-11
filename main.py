from drafter import *
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist

#Dataclasses
@dataclass
class Material:
    """Stores the physical properties of one metal or alloy."""
    name: str
    density: float
    youngs_modulus: float
    yield_strength: float


@dataclass
class State:
    """Stores all information that needs to persist across pages."""
    materials: list[Material]
    density_input: float
    modulus_input: float
    yield_input: float
    matched_materials: list[Material]
    comparison_scores: dict[str, float]
    status_message: str

#Material Database
def create_material_database() -> list[Material]:
    """Creates the reference database containing 15 materials."""
    return [
        Material("Aluminum 6061-T6", 2.70, 69.0, 276),
        Material("Aluminum 7075-T6", 2.81, 71.7, 503),
        Material("Aluminum 2024-T3", 2.78, 73.1, 345),
        Material("AISI 1018 Steel", 7.87, 205, 370),
        Material("AISI 1045 Steel", 7.87, 205, 310),
        Material("AISI 4140 Steel", 7.85, 205, 655),
        Material("A36 Structural Steel", 7.85, 200, 250),
        Material("Stainless Steel 304", 8.00, 193, 215),
        Material("Stainless Steel 316", 8.00, 193, 205),
        Material("Copper C110", 8.89, 117, 70),
        Material("Brass C260", 8.53, 110, 95),
        Material("Bearing Bronze", 8.80, 110, 125),
        Material("Titanium Ti-6Al-4V", 4.43, 113.8, 880),
        Material("Magnesium AZ31B", 1.77, 45, 200),
        Material("Nickel 200", 8.89, 204, 148)
    ]

#Helper Functions
def calculate_matches(state: State) -> None:
    """Compares user measurements against every database material."""

    user_values = np.array([[
        state.density_input,
        state.modulus_input,
        state.yield_input
    ]])

    material_values = []
    for material in state.materials:
        material_values.append([
            material.density,
            material.youngs_modulus,
            material.yield_strength
        ])
    material_values = np.array(material_values)

    scales = np.ptp(material_values, axis=0)

    distances = cdist(
        user_values / scales,
        material_values / scales,
        metric="euclidean"
    )[0]

    state.comparison_scores = {}
    for index in range(len(state.materials)):
        name = state.materials[index].name
        state.comparison_scores[name] = float(distances[index])

    closest_indices = np.argsort(distances)[:2]
    state.matched_materials = []
    for index in closest_indices:
        state.matched_materials.append(state.materials[index])

    if distances[closest_indices[0]] < 0.10:
        state.status_message = "A close material match was found!"
    else:
        state.status_message = (
            "The closest match may not be reliable. "
            "Check the comparison values."
        )

#Home Page
@route
def index(state: State) -> Page:
    """Displays the homepage and navigation options."""
    return Page(state, [
        Header("Material Strength Explorer"),
        "Identify an unknown metal using its physical properties.",
        HorizontalRule(),
        "Enter three measurements to find the closest "
        "matching metal or alloy in our database.",
        Button("Identify a Material", "input_page"),
        Button("Browse Material Database", "database_page"),
        Button("About This Project", "about_page")
    ])

#Input Page
@route
def input_page(state: State) -> Page:
    """Displays the form for entering material properties."""
    return Page(state, [
        Header("Material Identification"),
        "Enter the measured properties of your unknown material.",
        HorizontalRule(),
        "Density (g/cm^3):",
        TextBox("density", 2.70, kind="number"),
        "\n",
        "Young's Modulus (GPa):",
        TextBox("modulus", 69.0, kind="number"),
        "\n",
        "Yield Strength (MPa):",
        TextBox("yield_strength", 276, kind="number"),
        "\n",
        Button("Identify Material", "submit_material_properties"),
        Button("Home", "index")
    ])

#Submit and Process Input
@route
def submit_material_properties(
    state: State,
    density: float,
    modulus: float,
    yield_strength: float
) -> Page:
    """Validate user measurements and calculate material matches."""

    if density <= 0 or modulus <= 0 or yield_strength <= 0:
        state.status_message = "All measurements must be positive."
        return input_page(state)

    if density > 25 or modulus > 600 or yield_strength > 3000:
        state.status_message = "One or more measurements are unrealistic."
        return input_page(state)

    state.density_input = density
    state.modulus_input = modulus
    state.yield_input = yield_strength

    calculate_matches(state)
    return results_page(state)

#Results Page
@route
def results_page(state: State) -> Page:
    """Displays the closest matches and comparison information."""
    if len(state.matched_materials) == 0:
        return Page(state, [
            Header("No Results Yet"),
            "Please enter material measurements first.",
            Button("Enter Measurements", "input_page")
        ])

    best_match = state.matched_materials[0]
    second_match = state.matched_materials[1]

    return Page(state, [
        Header("Material Identification Results"),
        HorizontalRule(),
        "Your measurements:",
        "Density: " + str(state.density_input) + " g/cm^3",
        "Young's Modulus: " + str(state.modulus_input) + " GPa",
        "Yield Strength: " + str(state.yield_input) + " MPa",
        HorizontalRule(),
        Header("Best Match", 2),
        best_match.name,
        state.status_message,
        Header("Second Closest Match", 2),
        second_match.name,
        HorizontalRule(),
        "Comparison scores (smaller is better):",
        "Best match: " + str(
            state.comparison_scores[best_match.name]
        ),
        "Second match: " + str(
            state.comparison_scores[second_match.name]
        ),
        Button("View Full Database", "database_page"),
        Button("Try Again", "reset_inputs"),
        Button("Home", "index")
    ])

#Database Page
@route
def database_page(state: State) -> Page:
    """Displays all materials in the reference database."""
    content = [
        Header("Material Database"),
        "Our database currently contains 15 reference materials.",
        HorizontalRule()
    ]
    for material in state.materials:
        content.append(material.name)
        content.append(
            "Density: " + str(material.density) + " g/cm^3"
        )
        content.append(
            "Young's Modulus: " +
            str(material.youngs_modulus) + " GPa"
        )
        content.append(
            "Yield Strength: " +
            str(material.yield_strength) + " MPa"
        )
        content.append(HorizontalRule())
    content.append("Select a material to view its details:")
    content.append(
        SelectBox(
            "material_name",
            [material.name for material in state.materials]
        )
    )
    content.append(Button("View Material", "material_detail_page"))
    content.append(Button("Home", "index"))
    return Page(state, content)

#Material Details Page
@route
def material_detail_page(
    state: State,
    material_name: str
) -> Page:
    """Displays the properties of one selected material."""
    for material in state.materials:
        if material.name == material_name:
            return Page(state, [
                Header(material.name),
                HorizontalRule(),
                "Density: " + str(material.density) + " g/cm^3",
                "Young's Modulus: " +
                str(material.youngs_modulus) + " GPa",
                "Yield Strength: " +
                str(material.yield_strength) + " MPa",

                Button("Back to Database", "database_page"),
                Button("Home", "index")
            ])
    return Page(state, [
        "Material not found.",
        Button("Back to Database", "database_page")
    ])

#About Page
@route
def about_page(state: State) -> Page:
    """Explain the purpose and limitations of the application."""
    return Page(state, [
        Header("About Material Strength Explorer"),
        HorizontalRule(),
        "This application identifies unknown metals and alloys "
        "by comparing three material properties.",
        "The three properties are density, Young's modulus, "
        "and yield strength.",
        "NumPy organizes and processes the numerical data, "
        "while SciPy calculates the distance between the "
        "user's measurements and the reference materials.",
        "The application only provides estimated matches. "
        "Materials with similar properties may be difficult "
        "to distinguish using these three measurements alone.",
        Button("Home", "index"),
        Button("Identify a Material", "input_page")
    ])

#Reset
@route
def reset_inputs(state: State) -> Page:
    """Clears previous measurements and matching results."""
    state.density_input = 0.0
    state.modulus_input = 0.0
    state.yield_input = 0.0
    state.matched_materials = []
    state.comparison_scores = {}
    state.status_message = ""
    return input_page(state)

#Drafter Start
start_server(State(
    materials=create_material_database(),
    density_input=0.0,
    modulus_input=0.0,
    yield_input=0.0,
    matched_materials=[],
    comparison_scores={},
    status_message=""
))
