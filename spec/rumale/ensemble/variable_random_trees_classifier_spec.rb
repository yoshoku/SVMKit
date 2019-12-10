# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::VariableRandomTreesClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:n_estimators) { 10 }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(n_estimators: n_estimators, max_depth: 2, max_features: 2, n_jobs: n_jobs, random_seed: 1).fit(x, y)
  end
  let(:score) { estimator.score(x, y) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    it 'classifies two clusters data.', :aggregate_failures do
      expect(estimator.params[:n_estimators]).to eq(n_estimators)
      expect(estimator.params[:max_depth]).to eq(2)
      expect(estimator.params[:max_features]).to eq(2)
      expect(estimator.estimators.class).to eq(Array)
      expect(estimator.estimators.size).to eq(n_estimators)
      expect(estimator.estimators[0].class).to eq(Rumale::Tree::VariableRandomTreeClassifier)
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
    let(:probs) { estimator.predict_proba(x) }
    let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
    let(:index_mat) { estimator.apply(x) }
    let(:copied) { Marshal.load(Marshal.dump(estimator)) }

    it 'classifies three clusters data.', :aggregate_failures do
      expect(estimator.estimators.class).to eq(Array)
      expect(estimator.estimators.size).to eq(n_estimators)
      expect(estimator.estimators[0].class).to eq(Rumale::Tree::VariableRandomTreeClassifier)
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

    it 'returns leaf index that each sample reached', :aggregate_failures do
      expect(index_mat.ndim).to eq(2)
      expect(index_mat.shape[0]).to eq(n_samples)
      expect(index_mat.shape[1]).to eq(n_estimators)
      expect(index_mat[true, 0]).to eq(estimator.estimators[0].apply(x))
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params).to eq(copied.params)
      expect(estimator.estimators.size).to eq(copied.estimators.size)
      expect(estimator.classes).to eq(copied.classes)
      expect(estimator.feature_importances).to eq(copied.feature_importances)
      expect(estimator.rng).to eq(copied.rng)
      expect(score).to eq(copied.score(x, y))
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'classifies three clusters data in parallel.', :aggregate_failures do
        expect(estimator.estimators.class).to eq(Array)
        expect(estimator.estimators.size).to eq(n_estimators)
        expect(estimator.estimators[0].class).to eq(Rumale::Tree::VariableRandomTreeClassifier)
        expect(estimator.classes.class).to eq(Numo::Int32)
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.feature_importances.class).to eq(Numo::DFloat)
        expect(estimator.feature_importances.ndim).to eq(1)
        expect(estimator.feature_importances.shape[0]).to eq(n_features)
        expect(score).to eq(1.0)
      end
    end
  end
end
