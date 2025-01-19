#include <pybind11/pybind11.h>

namespace py = pybind11;

// A simple C++ function to add two numbers
int add(int a, int b) {
    return a + b;
}

// A C++ function that greets a user
std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

class Pet {
public:
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
private:
    std::string name;
};

// Binding code
PYBIND11_MODULE(example, m) {
    m.doc() = "Pybind11 example plugin"; // Optional module docstring
    m.def("add", &add, "A function to add two numbers",
          py::arg("a"), py::arg("b")); // Add argument names
    m.def("greet", &greet, "A function to greet a user",
          py::arg("name"));
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def_property("name", &Pet::getName, &Pet::setName)
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
}
