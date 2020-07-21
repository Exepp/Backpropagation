#include <blaze/Blaze.h>
#include <cstddef>

using Size_t = std::size_t;
using Real_t = float;


struct SigmoidActivFn {
    static Real_t Activation(Real_t z) { return 1.f / (1.f + blaze::exp(-z)); }
    static Real_t Derivative(Real_t z)
    {
        Real_t sVal = Activation(z);
        return sVal * (1.f * (1.f - sVal));
    }
};


struct Layer {
    using Vec_t = blaze::DynamicVector<Real_t, true>;
    using Weights_t = blaze::DynamicMatrix<Real_t>;
    using Biases_t = Vec_t;
    using ActivFn_t = Real_t (*)(Real_t);
    using ActivFnDeriv_t = ActivFn_t;

    Layer(Size_t prevLayerSize, Size_t thisLayerSize, ActivFn_t aFn = nullptr, ActivFnDeriv_t aFnD = nullptr);

    template <typename ActFn_t = void>
    void setActivFn()
    {
        if constexpr (!std::is_same_v<ActFn_t, void>) {
            activFn = ActFn_t::Activation;
            activFnDeriv = ActFn_t::Derivative;
        }
        else {
            activFn = nullptr;
            activFnDeriv = nullptr;
        }
    }

    std::pair<Layer::Vec_t, Layer::Vec_t> eval(Vec_t const& input) const; // {f(z), z}
    Vec_t evalDerivZ(Vec_t const& z) const;

    Weights_t weights;
    Biases_t biases;
    ActivFn_t activFn;
    ActivFnDeriv_t activFnDeriv;
};


class DenseNN {
public:
    DenseNN() = default;

    template <typename ActivFn_t = void>
    DenseNN& addLayer(Size_t size, Size_t inputSize = Size_t(-1))
    {
        if (!layers.size() && inputSize == Size_t(-1))
            throw std::logic_error("First layer must provide a valid inputSize");

        if (layers.size())
            layers.emplace_back(layers.back().biases.size(), size);
        else
            layers.emplace_back(inputSize, size);
        layers.back().setActivFn<ActivFn_t>();
        return *this;
    }

    std::pair<std::vector<Layer::Weights_t>, std::vector<Layer::Vec_t>>
    gradient(Layer::Vec_t const& input, Layer::Vec_t const& targetOutput) const;

    Layer::Vec_t operator()(Layer::Vec_t input) const;

    void learn(Layer::Vec_t const& input, Layer::Vec_t const& targetOutput);

private:
    std::vector<Layer> layers;
};