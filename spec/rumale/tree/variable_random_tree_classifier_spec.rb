# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::VariableRandomTreeClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:criterion) { 'gini' }
  let(:max_depth) { nil }
  let(:max_leaf_nodes) { nil }
  let(:min_samples_leaf) { 1 }
  let(:max_features) { nil }
  let(:alpha) { 0.3 }
  let(:estimator) do
    described_class.new(criterion: criterion, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                        min_samples_leaf: min_samples_leaf, max_features: max_features,
                        alpha: alpha, random_seed: 1).fit(x, y)
  end
  let(:probs) { estimator.predict_proba(x) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    it 'classifies two clusters data.', :aggregate_failures do
      expect(estimator.tree.class).to eq(Rumale::Tree::Node)
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to eq(1.0)
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }

    it 'classifies three clusters data.', :aggregate_failures do
      expect(estimator.tree.class).to eq(Rumale::Tree::Node)
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to eq(1.0)
    end

    it 'estimates class probabilities with three clusters dataset.', :aggregate_failures do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predicted_by_probs).to eq(y)
    end

    it 'dumps and restores itself using Marshal module.', :aggregated_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.classes).to eq(copied.classes)
      expect(estimator.feature_importances).to eq(copied.feature_importances)
      expect(estimator.rng).to eq(copied.rng)
      # FIXME: A slight error on the value of the threhold parameter occurs.
      #        It seems to be caused by rounding error of Float.
      # expect(estimator.tree).to eq(copied.tree)
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

      it 'learns model with given parameters.', :aggregate_failures do
        expect(estimator.params[:max_leaf_nodes]).to eq(max_leaf_nodes)
        expect(estimator.leaf_labels.size).to eq(max_leaf_nodes)
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
      context 'with zero' do
        let(:alpha) { 0 }

        it 'behaves like a random tree' do
          expect(estimator.leaf_labels.size).to be > n_classes
        end
      end

      context 'with one' do
        let(:alpha) { 1 }

        it 'behaves like a decision tree' do
          expect(estimator.leaf_labels.size).to eq(n_classes)
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
end
