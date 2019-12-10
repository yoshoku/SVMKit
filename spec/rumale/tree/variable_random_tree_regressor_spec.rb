# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::VariableRandomTreeRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x[true, 0] + x[true, 1]**2 }
  let(:multi_target) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.8, 0.2]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { multi_target.shape[1] }
  let(:criterion) { 'mse' }
  let(:max_depth) { nil }
  let(:max_leaf_nodes) { nil }
  let(:min_samples_leaf) { 1 }
  let(:max_features) { nil }
  let(:alpha) { 0.3 }
  let(:estimator) do
    described_class.new(criterion: criterion, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                        min_samples_leaf: min_samples_leaf, max_features: max_features,
                        alpha: alpha, random_seed: 8).fit(x, y)
  end
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when single target problem' do
    let(:y) { single_target }

    it 'learns the model for single regression problem.', :aggregate_failures do
      expect(estimator.tree.class).to eq(Rumale::Tree::Node)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(estimator.leaf_values.class).to eq(Numo::DFloat)
      expect(estimator.leaf_values.ndim).to eq(1)
      expect(estimator.leaf_values.shape[0]).not_to be_zero
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.tree.class).to eq(copied.tree.class)
      expect(estimator.feature_importances).to eq(copied.feature_importances)
      expect(estimator.rng).to eq(copied.rng)
      expect(score).to eq(copied.score(x, y))
    end

    context 'when max_depth parameter is given' do
      let(:max_depth) { 1 }

      it 'learns model with given parameters.', :aggregate_failures do
        expect(estimator.params[:max_depth]).to eq(max_depth)
        expect(estimator.tree.left.left).to be_nil
        expect(estimator.tree.left.right).to be_nil
        expect(estimator.tree.right.left).to be_nil
        expect(estimator.tree.right.right).to be_nil
      end
    end

    context 'when max_leaf_nodes parameter is given' do
      let(:max_leaf_nodes) { 2 }

      it 'learns model with given parameters.', :aggregate_failrues do
        expect(estimator.params[:max_leaf_nodes]).to eq(max_leaf_nodes)
        expect(estimator.leaf_values.size).to eq(max_leaf_nodes)
      end
    end

    context 'when min_samples_leaf parameter is given' do
      let(:min_samples_leaf) { 150 }

      it 'learns model with given parameters.', :aggregate_failures do
        expect(estimator.params[:min_samples_leaf]).to eq(min_samples_leaf)
        expect(estimator.tree.left.leaf).to be_truthy
        expect(estimator.tree.left.n_samples).to be >= min_samples_leaf
        expect(estimator.tree.right).to be_nil
      end
    end

    context 'when alpha parameter is given' do
      let(:max_depth) { 2 }

      context 'with zero' do
        let(:alpha) { 0 }

        it 'behaves like a random tree' do
          expect(score).to be < 0.5
        end
      end

      context 'with one' do
        let(:alpha) { 1 }

        it 'behaves like a decision tree' do
          expect(score).to be > 0.5
        end
      end
    end

    context 'when max_features parameter is given' do
      context 'with negative value' do
        let(:max_features) { -10 }

        it 'raises ArgumentError by validation' do
          expect { estimator }.to raise_error(ArgumentError)
        end
      end

      context 'with value larger than number of features' do
        let(:max_features) { 10 }

        it 'value of max_features is equal to the number of features' do
          expect(estimator.params[:max_features]).to eq(x.shape[1])
        end
      end

      context 'with valid value' do
        let(:max_features) { 2 }

        it 'learns model with given parameters.' do
          expect(estimator.params[:max_features]).to eq(2)
        end
      end
    end
  end

  context 'when multi-target problem' do
    let(:criterion) { 'mae' }
    let(:y) { multi_target }

    it 'learns the model for multiple regression problem.', :aggregate_failures do
      expect(estimator.tree.class).to eq(Rumale::Tree::Node)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(estimator.leaf_values.class).to eq(Numo::DFloat)
      expect(estimator.leaf_values.ndim).to eq(2)
      expect(estimator.leaf_values.shape[0]).not_to be_zero
      expect(estimator.leaf_values.shape[1]).to eq(n_outputs)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.01).of(1.0)
    end
  end
end
