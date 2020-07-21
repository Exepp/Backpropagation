#include <NeuralNetwork.hpp>
#include <iostream>

Layer::Layer(Size_t prevLayerSize, Size_t thisLayerSize, ActivFn_t aFn, ActivFnDeriv_t aFnD)
    : weights(prevLayerSize, thisLayerSize),
      biases(thisLayerSize),
      activFn(aFn),
      activFnDeriv(aFnD)
{
    weights = blaze::map(weights, [](Real_t x) { return blaze::rand<Real_t>() * 2 - 1; });
}

std::pair<Layer::Vec_t, Layer::Vec_t> Layer::eval(Layer::Vec_t const& input) const
{
    Vec_t zVec = input * weights + biases;

    return activFn ? std::pair(blaze::map(zVec, [this](Real_t z) { return activFn(z); }), zVec)
                   : std::pair(zVec, zVec);
}

Layer::Vec_t Layer::evalDerivZ(Layer::Vec_t const& z) const
{
    return activFnDeriv ? blaze::map(z, [this](Real_t z) { return activFnDeriv(z); })
                        : Vec_t(biases.size(), 1);
}

Layer::Vec_t DenseNN::operator()(Layer::Vec_t input) const
{
    for (auto const& layer : layers)
        // std::cout <<
        (input = layer.eval(input).first)
            // << '\n'
            ;
    return input;
}

std::pair<std::vector<Layer::Weights_t>, std::vector<Layer::Vec_t>>
DenseNN::gradient(Layer::Vec_t const& input, Layer::Vec_t const& targetOutput) const
{
    Size_t const S = layers.size();
    std::vector<Layer::Vec_t> zDerivatives(S);
    std::vector<Layer::Vec_t> zValues;
    std::vector<Layer::Vec_t> fzValues;
    Layer::Vec_t prev = input;

    for (auto const& layer : layers) {
        auto [fz, z] = layer.eval(prev);
        zValues.push_back(z);
        fzValues.push_back(prev = fz);
    }

    std::vector<Layer::Weights_t> wDerivatives(S);
    zDerivatives.back() = Real_t(2) / S * ((fzValues.back() - targetOutput) * layers.back().evalDerivZ(zValues.back()));
    for (std::ptrdiff_t s = S - 2; s >= 0; --s)
        zDerivatives[s] = blaze::trans(layers[s + 1].weights * blaze::trans(zDerivatives[s + 1])) * layers[s].evalDerivZ(zValues[s]);
    for (std::ptrdiff_t s = S - 1; s > 0; --s)
        wDerivatives[s] = blaze::trans(fzValues[s - 1]) * zDerivatives[s];
    wDerivatives.front() = blaze::trans(input) * zDerivatives.front();
    return std::pair(wDerivatives, zDerivatives);
}

void DenseNN::learn(Layer::Vec_t const& input, Layer::Vec_t const& targetOutput)
{
    auto [wGradient, bGradient] = gradient(input, targetOutput);

    for (std::ptrdiff_t s = 0; s < layers.size(); ++s) {
        layers[s].weights -= 0.1 * wGradient[s];
        layers[s].biases -= 0.1 * bGradient[s];
    }
}