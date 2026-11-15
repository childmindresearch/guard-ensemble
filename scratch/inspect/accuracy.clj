(ns accuracy
  (:require [cheshire.core :as json]
            [babashka.fs :as fs]
            [clojure.set :as set]
            [clojure.string :as str]
            [clojure.data.csv :as csv]))

;; Constants -------------------------------------------------------------------

(def ROOT
  (fs/canonicalize (fs/path (fs/parent *file*) ".." "..")))

(def OUT_FILES
  (->> (fs/path ROOT "out")
       (fs/list-dir)
       (map #(vector (fs/strip-ext (fs/file-name %)) %))
       (into {})))

(def TIERS [:green :yellow :red])

(def RANK (zipmap TIERS (range)))

;; Lib -------------------------------------------------------------------------

(defn read-jsonl [path]
  (->> (slurp (str path))
       (str/split-lines)
       (remove str/blank?)
       (map #(json/parse-string % keyword))))

(defn max-tier [tiers]
  (reduce (partial max-key #(get RANK % 1)) :green tiers))

(defn concern? [{:keys [entry_tier] :as _entry}]
  (not= "green" entry_tier))

(defn high-risk? [{:keys [entry_tier] :as _entry}]
  (= "red" entry_tier))

(defn float->str [x]
  (format "%.3f" (double x)))

(defn unify-tier [m [[lbl-tier-val ent-tier-val] n]]
  (assoc-in m [(keyword lbl-tier-val) (keyword ent-tier-val)] n))

(defn score-tiers [label-tier entries]
  (->> entries
       (filter label-tier)
       (map (juxt label-tier :entry_tier))
       (frequencies)
       (reduce unify-tier {})
       (map #(hash-map :labeled (key %) :output (val %)))
       (sort-by #(get RANK (:labeled %)))
       (into [])))

;; Label scores

(defn count-by [pred coll]
  (count (filter pred coll)))

(def score-label-preds
  {:high_risk_conservative high-risk?
   :concern_conservative   concern?
   :high_risk_liberal      high-risk?
   :concern_liberal        concern?})

(defn score-label [label-k entries]
  (when-let [pred (get score-label-preds label-k)]
    (let [rated     (filter (comp boolean? label-k) entries)
          true-pos  (count-by #(and (label-k %) (pred %)) rated)
          false-neg (count-by #(and (label-k %) (not (pred %))) rated)
          false-pos (count-by #(and (not (label-k %)) (pred %)) rated)
          true-neg  (count-by #(and (not (label-k %)) (not (pred %))) rated)]
      {:recall    (float (/ true-pos (max 1 (+ true-pos false-neg))))
       :precision (float (/ true-pos (max 1 (+ true-pos false-pos))))
       :confusion {:true-pos  true-pos
                   :false-pos false-pos
                   :false-neg false-neg
                   :true-neg  true-neg}})))

(defn consensus [con-k lib-k entries]
  (filter #(and (some? (con-k %)) (= (con-k %) (lib-k %))) entries))

;; Run health: unparsed domains, error kinds, confidence.

;; Reports --------------------------------------------------------------------

(defn derive-scores [entries]
  (let [cns-risk (consensus :high_risk_conservative :high_risk_liberal entries)
        cns-conc (consensus :concern_conservative :concern_liberal entries)
        cns-isec (vec (set/intersection (set cns-risk) (set cns-conc)))]
    {:conservative {:high-risk (score-label :high_risk_conservative entries)
                    :concern   (score-label :concern_conservative entries)
                    :tiers     (score-tiers :tier_conservative entries)}
     :liberal      {:high-risk (score-label :high_risk_liberal entries)
                    :concern   (score-label :concern_liberal entries)
                    :tiers     (score-tiers :tier_liberal entries)}
     :consensus    {:high-risk (score-label :high_risk_conservative cns-risk)
                    :concern   (score-label :concern_conservative cns-conc)
                    :tiers     (score-tiers :tier_liberal cns-isec)}}))

(def scores->row-schema
  [["group"]
   ["high_risk_recall"       [:high-risk :recall]]
   ["high_risk_precision"    [:high-risk :precision]]
   ["high_risk_confusion_tp" [:high-risk :confusion :true-pos]]
   ["high_risk_confusion_fp" [:high-risk :confusion :false-pos]]
   ["high_risk_confusion_tn" [:high-risk :confusion :true-neg]]
   ["high_risk_confusion_fn" [:high-risk :confusion :false-neg]]
   ["concern_recall"         [:concern :recall]]
   ["concern_precision"      [:concern :precision]]
   ["concern_confusion_tp"   [:concern :confusion :true-pos]]
   ["concern_confusion_fp"   [:concern :confusion :false-pos]]
   ["concern_confusion_tn"   [:concern :confusion :true-neg]]
   ["concern_confusion_fn"   [:concern :confusion :false-neg]]
   ["tier_green_to_green"    [:tiers 0 :output :green]]
   ["tier_green_to_yellow"   [:tiers 0 :output :yellow]]
   ["tier_green_to_red"      [:tiers 0 :output :red]]
   ["tier_yellow_to_green"   [:tiers 1 :output :green]]
   ["tier_yellow_to_yellow"  [:tiers 1 :output :yellow]]
   ["tier_yellow_to_red"     [:tiers 1 :output :red]]
   ["tier_red_to_green"      [:tiers 2 :output :green]]
   ["tier_red_to_yellow"     [:tiers 2 :output :yellow]]
   ["tier_red_to_red"        [:tiers 2 :output :red]]])

(defn scores->row [[group scores]]
  (->> scores->row-schema
       (drop 1)
       (mapv (comp (partial get-in scores) second))
       (concat [(name group)])
       (into [])))

(defn report->csv [data]
  (let [header (mapv first scores->row-schema)
        str-w  (java.io.StringWriter.)]
    (->> data
         (map scores->row)
         (concat [header])
         (csv/write-csv str-w))
    (str str-w)))

;; Drill-down: agreed-red entries the model didn't call red, with Harm to Self evidence.

(defn missed-reds [entries]
  (->> (consensus :high_risk_conservative :high_risk_liberal entries)
       (filter #(and (true? (:high_risk_conservative %)) (not (high-risk? %))))
       (map #(-> %
                 (select-keys [:id :text :entry_tier :entry_reasons])
                 (assoc :harm_to_self (get-in % [:domain_results :01_harm_to_self :evidence]))))))

(defn false-reds [entries]
  (->> (consensus :high_risk_conservative :high_risk_liberal entries)
       (filter #(and (false? (:high_risk_conservative %)) (high-risk? %)))
       (map #(-> %
                 (select-keys [:id :text :entry_reasons :entry_tier])
                 (assoc :red_domains (->> (:domain_results %)
                                          (filter (comp #{"red"} :tier val))
                                          (into {} (map (juxt key (comp :evidence val))))))))))

(defn ai-risk [entries]
  (let [path [:domain_results :04_ai_related_safety_risk :tier]]
    (filter #(not= "green" (get-in % path)) entries)))

;; Variables -------------------------------------------------------------------

;; (def $output-stem "key_domains_qwen3_4b_full_0002")
;; (def $output-stem "key_domains_qwen3_4b_2507_labelled_0002")
;; (def $output-stem "key_domains_qwen3_4b_2507_labelled_0003")
(def $output-stem "key_domains_qwen3_4b_2507_labelled_0004")

;; Pipeline --------------------------------------------------------------------

(def entries (read-jsonl (get OUT_FILES $output-stem)))

(def report (derive-scores entries))

report

(comment
  (report->csv report)

  (count (ai-risk entries))

  (take 10 (missed-reds entries)))
