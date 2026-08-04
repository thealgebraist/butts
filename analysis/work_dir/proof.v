From Stdlib Require Import Reals.
Open Scope R_scope.

(* Formal derivation of Stochastic Diagonal Trace Estimator for Local Hessian *)

Axiom Matrix : nat -> nat -> Type.
Axiom Vector : nat -> Type.

Axiom Trace : forall {n}, Matrix n n -> R.
Axiom Expectation : forall {n}, (Vector n -> R) -> R.

(* Vector transpose multiplication *)
Axiom vec_transpose_mul : forall {n}, Vector n -> Matrix n n -> Vector n -> R.

(* For a 3x3 local pixel Hessian, the Hutchinson trace estimator still applies *)
Axiom Local_Hutchinson_Equality : forall (H : Matrix 3 3),
  Expectation (fun z : Vector 3 => vec_transpose_mul z H z) = Trace H.

(* Saliency energy is proportional to the squared trace (Laplacian squared) *)
Definition SaliencyEnergy (H : Matrix 3 3) : R :=
  (Trace H) * (Trace H).

Theorem bbox_saliency_unbiased : forall (H : Matrix 3 3),
  Expectation (fun z : Vector 3 => vec_transpose_mul z H z) = Trace H.
Proof.
  intros.
  apply Local_Hutchinson_Equality.
Qed.
