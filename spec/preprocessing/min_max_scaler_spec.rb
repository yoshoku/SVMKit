require 'spec_helper'

RSpec.describe SVMKit::Preprocessing::MinMaxScaler do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) do
    rng = Random.new(1)
    rnd_vals = Array.new(n_samples * n_features) { rng.rand }
    NMatrix.new([n_samples, n_features], rnd_vals, dtype: :float64, stype: :dense)
  end

  it 'normalizes range of features to [0,1].' do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    expect(normalized.min.to_a.min).to eq(0)
    expect(normalized.max.to_a.max).to eq(1)
  end

  it 'normalizes range of features to a given range.' do
    normalizer = described_class.new(feature_range: [-3, 2])
    normalized = normalizer.fit_transform(samples)
    expect(normalized.min.to_a.min).to eq(-3)
    expect(normalized.max.to_a.max).to eq(2)
  end

  it 'dumps and restores itself using Marshal module.' do
    transformer = described_class.new
    transformer.fit(samples)
    copied = Marshal.load(Marshal.dump(transformer))
    expect(transformer.min_vec).to eq(copied.min_vec)
    expect(transformer.max_vec).to eq(copied.max_vec)
    expect(transformer.params[:feature_range][0]).to eq(copied.params[:feature_range][0])
    expect(transformer.params[:feature_range][1]).to eq(copied.params[:feature_range][1])
  end
end