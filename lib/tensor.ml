open! Core

type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t

let dims (t : t) = Bigarray.Genarray.dims t
let num_dims (t : t) = Bigarray.Genarray.num_dims t
let length (t : t) = dims t |> Array.fold ~init:1 ~f:( * )

let item (t : t) =
  match dims t with
  | [||] -> Bigarray.Genarray.get t [||]
  | dims -> raise_s [%message "Tensor.item: dims > 0" (dims : int array)]
;;

let get (t : t) index = Bigarray.Genarray.get t index
let set (t : t) index value = Bigarray.Genarray.set t index value
let fill (t : t) value = Bigarray.Genarray.fill t value

let rec sexp_of_t t =
  match dims t with
  | [||] -> item t |> [%sexp_of: float]
  | [| n |] -> List.init n ~f:(fun i -> get t [| i |]) |> [%sexp_of: float list]
  | dims ->
    let first_dim = dims.(0) in
    List.init first_dim ~f:(fun i -> Bigarray.Genarray.slice_left t [| i |] |> sexp_of_t)
    |> [%sexp_of: Sexp.t list]
;;

let create_uninitialized dims =
  Bigarray.Genarray.create Bigarray.float64 Bigarray.c_layout dims
;;

let reshape t ~dims = Bigarray.reshape t dims

(* TODO: write a ppx that allows writing [5t] or [5.5t] which expands to
   [Tensor.of_float (Int.to_float 5)] and [Tensor.of_float 5.5]. See
   janestreet/ppx_fixed_literal for prior art. *)
let of_float f =
  let t = create_uninitialized [||] in
  set t [||] f;
  t
;;

let%expect_test "of_float" =
  let t = of_float 5. in
  [%sexp_of: t] t |> print_s;
  [%expect {| 5 |}]
;;

let create ~dims value =
  let t = create_uninitialized dims in
  fill t value;
  t
;;

let zeros ~dims = create ~dims 0.
let ones ~dims = create ~dims 1.

let arange n =
  let t = create_uninitialized [| n |] in
  for i = 0 to n - 1 do
    Bigarray.Genarray.set t [| i |] (Int.to_float i)
  done;
  t
;;

let%expect_test "arange" =
  arange 12 |> sexp_of_t |> print_s;
  [%expect {| (0 1 2 3 4 5 6 7 8 9 10 11) |}];
  arange 12 |> reshape ~dims:[| 6; 2 |] |> sexp_of_t |> print_s;
  [%expect {| ((0 1) (2 3) (4 5) (6 7) (8 9) (10 11)) |}];
  arange 12 |> reshape ~dims:[| 3; 4 |] |> sexp_of_t |> print_s;
  [%expect {| ((0 1 2 3) (4 5 6 7) (8 9 10 11)) |}]
;;

let of_xla_literal literal = Xla.Literal.to_bigarray literal ~kind:Bigarray.float64

let%expect_test "xla" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  let cpu = Xla.Client.cpu () in
  let builder = Xla.Builder.create ~name:"test_builder" in
  let root =
    let f = Xla.Op.r0_f64 ~builder in
    Xla.Op.add (f 39.) (f 3.)
  in
  let computation = Xla.Computation.build ~root in
  let exe = Xla.Executable.compile cpu computation in
  let buffers = Xla.Executable.execute exe [||] in
  let literal = Xla.Buffer.to_literal_sync buffers.(0).(0) in
  let t = of_xla_literal literal in
  print_s [%message "" (t : t)];
  [%expect {| (t 42) |}]
;;

let map t ~f =
  match num_dims t with
  | 0 -> get t [||] |> f |> of_float
  | _ ->
    let t' = create_uninitialized [| length t |] in
    for i = 0 to length t - 1 do
      set t' [| i |] (f (get t [| i |]))
    done;
    reshape t' ~dims:(dims t)
;;

let map2 t1 t2 ~f =
  let dims1 = dims t1 in
  let dims2 = dims t2 in
  if not ([%compare.equal: int array] dims1 dims2)
  then
    raise_s
      [%message "Tensor.map2: dims mismatch" (dims1 : int array) (dims2 : int array)];
  match num_dims t1 with
  | 0 -> f (get t1 [||]) (get t2 [||]) |> of_float
  | _ ->
    let t = create_uninitialized [| length t1 |] in
    for i = 0 to length t1 - 1 do
      set t [| i |] (f (get t1 [| i |]) (get t2 [| i |]))
    done;
    reshape t ~dims:(dims t1)
;;

module O = struct
  let ( + ) t1 t2 = map2 t1 t2 ~f:( +. )
  let ( - ) t1 t2 = map2 t1 t2 ~f:( -. )
  let ( * ) t1 t2 = map2 t1 t2 ~f:( *. )
  let ( ~- ) t = map t ~f:( ~-. )
end

include O

let sin t = map t ~f:Float.sin
let cos t = map t ~f:Float.cos
