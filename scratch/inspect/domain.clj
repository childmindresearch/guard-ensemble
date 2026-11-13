(ns domain
  (:require [cheshire.core :as json]
            [babashka.fs :as fs]
            [clojure.string :as str]))

;; Constants -------------------------------------------------------------------

(def OUT_DIR
  (fs/canonicalize (fs/path (fs/parent *file*) ".." ".." "out")))

(def OUT_FILES
  (->> OUT_DIR
       (fs/list-dir)
       (map #(vector (fs/strip-ext (fs/file-name %)) %))
       (into {})))

;; Lib -------------------------------------------------------------------------

(defn read-file [path]
  (slurp (str path)))

(defn parse-entries [raw]
  (map #(json/parse-string % keyword) (str/split raw #"\n")))

(defn domain-tiers [{:keys [domain_results] :as _entry}]
  (let [pick (juxt key (comp :tier val))]
    (set (map pick (seq domain_results)))))

(defn assoc-domain-tiers [entries]
  (map #(assoc % :domain_tiers (domain-tiers %)) entries))

(defn ai-risk [entries]
  (let [path [:domain_tiers [:04_ai_related_safety_risk "green"]]]
    (->> entries (assoc-domain-tiers) (remove #(get-in % path)))))

(defn sum-risk-confidence [{:keys [domain_results] :as _row}]
  (->> domain_results
       (vals)
       (remove (comp #{"green"} :tier))
       (map :confidence)
       (reduce + 0)))

(defn dedupe-text [entries]
  (let [max-conf (comp first (partial sort-by sum-risk-confidence >))]
    (->> entries (group-by :text) (vals) (map max-conf))))

(defn elevated-domain-results [domain_results]
  (into {} (filter (comp (partial not= "green") :tier val) domain_results)))

(defn summarize-domain-results [domain_results]
  (-> domain_results
      (elevated-domain-results)
      (update-vals #(select-keys % [:tier :evidence]))))

(defn summarize [entries]
  (map #(-> %
            (select-keys [:text :domain_results])
            (update :domain_results summarize-domain-results))
       entries))

;; Variables -------------------------------------------------------------------

;; (def $output-stem "key_domains_qwen3_4b_full_0001")
;; (def $output-stem "key_domains_qwen3_4b_2507_labelled_0001")
(def $output-stem "key_domains_qwen3_4b_2507_labelled_0002")

;; Pipeline --------------------------------------------------------------------

(defn inspect-output
  ([file & {summarize? :summarize ai-risk-only? :ai-risk-only}]
   (-> file
       (read-file)
       (parse-entries)
       (cond-> ai-risk-only? (ai-risk))
       (dedupe-text)
       (cond-> summarize? (summarize)))))

(def domain-entries
  (inspect-output (get OUT_FILES $output-stem) :summarize true :ai-risk-only true))

domain-entries

(count domain-entries)
