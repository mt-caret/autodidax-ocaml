open! Core

module type Interpreter_level = sig
  type global_data

  val level : int
  val global_data : global_data
end

module Shaped_array = struct
  type t = { dims : int array } [@@deriving sexp, compare]

  let of_tensor tensor = { dims = Tensor.dims tensor }
end

module Abstract_value = struct
  type t =
    | Shaped_array of Shaped_array.t
    | Concrete_array of Tensor.t
  [@@deriving sexp_of]

  let shaped_array = function
    | Shaped_array shaped_array -> shaped_array
    | Concrete_array t -> { dims = Tensor.dims t }
  ;;

  let dims = function
    | Shaped_array { dims } -> dims
    | Concrete_array t -> Tensor.dims t
  ;;
end

module type Tracer = sig
  type t [@@deriving sexp_of]
  type value

  val aval : t -> Abstract_value.t
  val full_lower : t -> value
end

module Var : sig
  module Id : sig
    include Comparable.S

    val to_int : t -> int
    val to_string : t -> string
  end

  type t [@@deriving sexp, compare]

  val create : Shaped_array.t -> t
  val shaped_array : t -> Shaped_array.t
  val id : t -> Id.t
  val lookup : Id.t -> t
end = struct
  module Id = struct
    include Id.Make (struct
        let name = "Jax.Var"
      end)

    let to_string t =
      let t = to_int t in
      [%string "var_%{t#Int}"]
    ;;
  end

  type t =
    { shaped_array : Shaped_array.t
    ; id : Id.t
    }
  [@@deriving sexp, compare]

  let store = ref Id.Map.empty

  let create shaped_array =
    let t = { shaped_array; id = Id.create () } in
    store := Map.add_exn !store ~key:t.id ~data:t;
    t
  ;;

  let shaped_array t = t.shaped_array
  let id t = t.id
  let lookup id = Map.find_exn !store id
end

module Atom = struct
  type t =
    | Var of Var.t
    | Lit of Tensor.t
  [@@deriving sexp_of, compare]

  let shaped_array = function
    | Var var -> Var.shaped_array var
    | Lit tensor -> Shaped_array.of_tensor tensor
  ;;
end

module rec Primitive : sig
  type t =
    | Sin
    | Cos
    | Neg
    | Add
    | Sub
    | Mul
    | Xla_call of
        { jaxpr : Jaxpr0.t
        ; num_consts : int
        }
  [@@deriving sexp_of, compare]

  val to_string : t -> string
end = struct
  type t =
    | Sin
    | Cos
    | Neg
    | Add
    | Sub
    | Mul
    | Xla_call of
        { jaxpr : Jaxpr0.t
        ; num_consts : int
        }
  [@@deriving sexp_of, compare]

  let to_string t = sexp_of_t t |> Sexp.to_string |> String.lowercase
end

and Jaxpr0 : sig
  module Eqn : sig
    type t =
      { prim : Primitive.t
      ; inputs : Atom.t Nonempty_list.t
      ; out_binders : Var.t Nonempty_list.t
      }
    [@@deriving sexp_of, compare]
  end

  module Type : sig
    type t =
      { in_types : Shaped_array.t Nonempty_list.t
      ; out_types : Shaped_array.t Nonempty_list.t
      }
    [@@deriving sexp_of]
  end

  type t =
    { in_binders : Var.t Nonempty_list.t
    ; eqns : Eqn.t list
    ; outs : Atom.t Nonempty_list.t
    }
  [@@deriving sexp_of, compare]
end = struct
  module Eqn = struct
    type t =
      { prim : Primitive.t
      ; inputs : Atom.t Nonempty_list.t
      ; out_binders : Var.t Nonempty_list.t
      }
    [@@deriving sexp_of, compare]
  end

  module Type = struct
    type t =
      { in_types : Shaped_array.t Nonempty_list.t
      ; out_types : Shaped_array.t Nonempty_list.t
      }
    [@@deriving sexp_of]
  end

  type t =
    { in_binders : Var.t Nonempty_list.t
    ; eqns : Eqn.t list
    ; outs : Atom.t Nonempty_list.t
    }
  [@@deriving sexp_of, compare]
end

module type Interpreter0 = sig
  type packed_tracer

  val name : string

  module Tracer : Tracer
  module Level : Interpreter_level

  val tracer_witness : Tracer.t Type_equal.Id.t
  val pure : Tensor.t -> Tracer.t
  val lift : packed_tracer -> Tracer.t

  val process_primitive
    :  Primitive.t
    -> Tracer.t Nonempty_list.t
    -> Tracer.t Nonempty_list.t
end

module rec Interpreter : sig
  type ('tracer, 'global_data) t =
    (module Interpreter0
       with type Tracer.t = 'tracer
        and type Tracer.value = Value.t
        and type Level.global_data = 'global_data
        and type packed_tracer = Packed_tracer.t)
end = struct
  type ('tracer, 'global_data) t =
    (module Interpreter0
       with type Tracer.t = 'tracer
        and type Tracer.value = Value.t
        and type Level.global_data = 'global_data
        and type packed_tracer = Packed_tracer.t)
end

and Packed_interpreter : sig
  type t = T : ('tracer, 'global_data) Interpreter.t -> t
end = struct
  type t = T : ('tracer, 'global_data) Interpreter.t -> t
end

and Packed_tracer : sig
  type t =
    | T :
        { tracer : 'tracer
        ; interpreter : ('tracer, 'global_data) Interpreter.t
        }
        -> t
  [@@deriving sexp_of]
end = struct
  type t =
    | T :
        { tracer : 'tracer
        ; interpreter : (('tracer, 'global_data) Interpreter.t[@sexp.opaque])
        }
        -> t

  let sexp_of_t (T { tracer; interpreter = (module Interpreter) }) =
    [%sexp_of: string * Interpreter.Tracer.t] (Interpreter.name, tracer)
  ;;
end

and Value : sig
  type t [@@deriving sexp_of]

  module Id : Comparable.S

  module Hide_id : sig
    type nonrec t = t [@@deriving sexp_of]
  end

  val of_float : float -> t
  val of_tensor : Tensor.t -> t
  val of_tracer : Packed_tracer.t -> t
  val get : t -> [ `Tensor of Tensor.t | `Tracer of Packed_tracer.t ]
  val id : t -> Id.t
  val get_aval : t -> Abstract_value.t
  val full_lower : t -> t
  val full_raise : ('tracer, _) Interpreter.t -> t -> 'tracer
end = struct
  module Id = Id.Make (struct
      let name = "Jax.Value"
    end)

  type t =
    { id : Id.t
    ; value : [ `Tensor of Tensor.t | `Tracer of Packed_tracer.t ]
    }
  [@@deriving sexp_of]

  module Hide_id = struct
    type nonrec t = t

    let sexp_of_t t =
      [%sexp_of: [ `Tensor of Tensor.t | `Tracer of Packed_tracer.t ]] t.value
    ;;
  end

  let of_tensor tensor = { id = Id.create (); value = `Tensor tensor }
  let of_float f = of_tensor (Tensor.of_float f)
  let of_tracer tracer = { id = Id.create (); value = `Tracer tracer }
  let get t = t.value
  let id t = t.id

  let get_aval t =
    match get t with
    | `Tensor t -> Abstract_value.Concrete_array t
    | `Tracer (T { tracer; interpreter = (module Interpreter) }) ->
      Interpreter.Tracer.aval tracer
  ;;

  let full_lower t =
    match get t with
    | `Tensor _ -> t
    | `Tracer (T { tracer; interpreter = (module Interpreter) }) ->
      Interpreter.Tracer.full_lower tracer
  ;;

  let full_raise
    (type tracer global_data)
    ((module Interpreter) : (tracer, global_data) Interpreter.t)
    t
    : tracer
    =
    match get t with
    | `Tensor t -> Interpreter.pure t
    | `Tracer (T { tracer; interpreter = (module Interpreter_to_lift) } as packed_tracer)
      ->
      (match
         Ordering.of_int (compare Interpreter.Level.level Interpreter_to_lift.Level.level)
       with
       | Equal ->
         let T =
           Type_equal.Id.same_witness_exn
             Interpreter.tracer_witness
             Interpreter_to_lift.tracer_witness
         in
         tracer
       | Greater -> Interpreter.lift packed_tracer
       | Less ->
         raise_s
           [%message "Cannot lift a higher-level tracer to a lower-level interpreter"])
  ;;
end

let interpreter_stack : Packed_interpreter.t list ref = ref []
let dynamic_interpreter : Packed_interpreter.t option ref = ref None

let find_top_interpreter values =
  let top_interpreter =
    Nonempty_list.filter_map values ~f:(fun value ->
      match Value.get value with
      | `Tensor _ -> None
      | `Tracer packed_tracer -> Some packed_tracer)
    |> List.max_elt
         ~compare:
           (Comparable.lift
              Int.compare
              ~f:
                (fun
                  (Packed_tracer.T { tracer = _; interpreter = (module Interpreter) }) ->
                Interpreter.Level.level))
    |> Option.map ~f:(fun (Packed_tracer.T { tracer = _; interpreter }) ->
      Packed_interpreter.T interpreter)
    |> Option.value ~default:(List.last_exn !interpreter_stack)
  in
  match !dynamic_interpreter with
  | None -> top_interpreter
  | Some (Packed_interpreter.T (module Dynamic_interpreter) as dynamic_interpreter) ->
    let (Packed_interpreter.T (module Top_interpreter)) = top_interpreter in
    if Top_interpreter.Level.level > Dynamic_interpreter.Level.level
    then top_interpreter
    else dynamic_interpreter
;;

let new_interpreter
  (type tracer global_data)
  ~(create_interpreter : level:int -> global_data -> (tracer, global_data) Interpreter.t)
  ~global_data
  ~f
  =
  let level = List.length !interpreter_stack in
  let interpreter = create_interpreter ~level global_data in
  interpreter_stack := Packed_interpreter.T interpreter :: !interpreter_stack;
  protect
    ~f:(fun () -> f interpreter)
    ~finally:(fun () -> interpreter_stack := List.tl_exn !interpreter_stack)
;;

let new_dynamic_interpreter interpreter ~f =
  let old_dynamic_interpreter = !dynamic_interpreter in
  dynamic_interpreter := Some interpreter;
  protect
    ~f:(fun () -> f ())
    ~finally:(fun () -> dynamic_interpreter := old_dynamic_interpreter)
;;

let debug = ref false
let jaxpr_to_string = Set_once.create ()

let bind prim values =
  let (T interpreter) = find_top_interpreter values in
  let tracers =
    Nonempty_list.map values ~f:(fun value -> Value.full_raise interpreter value)
  in
  let (module Interpreter) = interpreter in
  if !debug
  then (
    print_s [%message "bind" (prim : Primitive.t) (values : Value.t Nonempty_list.t)];
    match prim with
    | Xla_call { jaxpr; num_consts = _ } ->
      Set_once.get_exn jaxpr_to_string [%here] jaxpr |> print_endline
    | _ -> ());
  Interpreter.process_primitive prim tracers
  |> Nonempty_list.map ~f:Interpreter.Tracer.full_lower
;;

let bind1 prim values =
  match bind prim values with
  | [ t ] -> t
  | _ -> raise_s [%message "Expected a single result" (prim : Primitive.t)]
;;

let ( + ) t1 t2 = bind1 Primitive.Add [ t1; t2 ]
let ( - ) t1 t2 = bind1 Primitive.Sub [ t1; t2 ]
let ( * ) t1 t2 = bind1 Primitive.Mul [ t1; t2 ]
let ( ~- ) t = bind1 Primitive.Neg [ t ]
let sin t = bind1 Primitive.Sin [ t ]
let cos t = bind1 Primitive.Cos [ t ]
let xla_callable' = Set_once.create ()

let eval_interpreter ~level : (Tensor.t, unit) Interpreter.t =
  (module struct
    type packed_tracer = Packed_tracer.t

    let name = "eval_interpreter"

    module Tracer = struct
      type t = Tensor.t [@@deriving sexp_of]
      type value = Value.t

      let aval t = Abstract_value.Concrete_array t
      let full_lower t = Value.of_tensor t
    end

    module Level = struct
      type global_data = unit

      let level = level
      let global_data = ()
    end

    let tracer_witness = Type_equal.Id.create ~name:"tensor" [%sexp_of: Tensor.t]
    let pure = Fn.id

    let lift packed_tracer =
      raise_s
        [%message "Cannot lift in eval interpreter" (packed_tracer : Packed_tracer.t)]
    ;;

    let process_primitive (prim : Primitive.t) (tracers : Tensor.t Nonempty_list.t)
      : Tensor.t Nonempty_list.t
      =
      match prim, tracers with
      | Sin, [ t ] -> [ Tensor.sin t ]
      | Cos, [ t ] -> [ Tensor.cos t ]
      | Neg, [ t ] -> [ Tensor.( ~- ) t ]
      | Add, [ t1; t2 ] -> [ Tensor.( + ) t1 t2 ]
      | Sub, [ t1; t2 ] -> [ Tensor.( - ) t1 t2 ]
      | Mul, [ t1; t2 ] -> [ Tensor.( * ) t1 t2 ]
      | Xla_call { jaxpr; num_consts }, args ->
        let consts, args = List.split_n (Nonempty_list.to_list args) num_consts in
        let callable =
          (Set_once.get_exn xla_callable' [%here]) jaxpr consts |> Staged.unstage
        in
        callable (Nonempty_list.of_list_exn args)
      | _ ->
        raise_s
          [%message
            "unexpected evaluation" (prim : Primitive.t) (tracers : _ Nonempty_list.t)]
    ;;
  end)
;;

let () = interpreter_stack := [ Packed_interpreter.T (eval_interpreter ~level:0) ]

let%expect_test "eval" =
  let f x =
    let y = sin x * Value.of_float 2. in
    let z = -y + x in
    z
  in
  Value.of_float 3. |> f |> [%sexp_of: Value.Hide_id.t] |> print_s;
  [%expect {| (Tensor 2.7177599838802657) |}]
;;

module Jvp_tracer = struct
  type t =
    { primal : Value.t
    ; tangent : Value.t
    }
  [@@deriving sexp_of]
end

let jvp_jaxpr' = Set_once.create ()

let jvp_interpreter ~level () : (Jvp_tracer.t, unit) Interpreter.t =
  let rec interpreter : (Jvp_tracer.t, unit) Interpreter.t lazy_t =
    lazy
      (module struct
        type packed_tracer = Packed_tracer.t

        let name = "jvp_interpreter"

        module Tracer = struct
          type t = Jvp_tracer.t [@@deriving sexp_of]
          type value = Value.t

          let aval { Jvp_tracer.primal; tangent = _ } = Value.get_aval primal

          let full_lower t =
            Value.of_tracer (T { tracer = t; interpreter = force interpreter })
          ;;
        end

        module Level = struct
          type global_data = unit

          let level = level
          let global_data = ()
        end

        let tracer_witness =
          Type_equal.Id.create ~name:"jvp_tracer" [%sexp_of: Jvp_tracer.t]
        ;;

        let value_to_tracer value =
          { Jvp_tracer.primal = value
          ; tangent =
              Value.of_tensor
                (Tensor.zeros ~dims:(Value.get_aval value |> Abstract_value.dims))
          }
        ;;

        let pure tensor = value_to_tracer (Value.of_tensor tensor)
        let lift packed_tracer = value_to_tracer (Value.of_tracer packed_tracer)

        let process_primitive
          (prim : Primitive.t)
          (tracers : Jvp_tracer.t Nonempty_list.t)
          : Jvp_tracer.t Nonempty_list.t
          =
          match prim, tracers with
          | Sin, [ { primal = x; tangent = dx } ] ->
            [ { primal = sin x; tangent = cos x * dx } ]
          | Cos, [ { primal = x; tangent = dx } ] ->
            [ { primal = cos x; tangent = -sin x * dx } ]
          | Neg, [ { primal = x; tangent = dx } ] -> [ { primal = -x; tangent = -dx } ]
          | Add, [ { primal = x1; tangent = dx1 }; { primal = x2; tangent = dx2 } ] ->
            [ { primal = x1 + x2; tangent = dx1 + dx2 } ]
          | Sub, [ { primal = x1; tangent = dx1 }; { primal = x2; tangent = dx2 } ] ->
            [ { primal = x1 - x2; tangent = dx1 - dx2 } ]
          | Mul, [ { primal = x1; tangent = dx1 }; { primal = x2; tangent = dx2 } ] ->
            [ { primal = x1 * x2; tangent = (x1 * dx2) + (dx1 * x2) } ]
          | Xla_call { jaxpr; num_consts = _ }, args ->
            let new_jaxpr, new_consts = (Set_once.get_exn jvp_jaxpr' [%here]) jaxpr in
            let primals, tangents =
              Nonempty_list.map args ~f:(fun { primal; tangent } -> primal, tangent)
              |> Nonempty_list.unzip
            in
            let outs =
              bind
                (Xla_call { jaxpr = new_jaxpr; num_consts = List.length new_consts })
                (Nonempty_list.append primals (Nonempty_list.to_list tangents))
            in
            List.split_n (Nonempty_list.to_list outs) (Nonempty_list.length outs / 2)
            |> Tuple2.uncurry List.zip_exn
            |> List.map ~f:(fun (primal, tangent) -> { Jvp_tracer.primal; tangent })
            |> Nonempty_list.of_list_exn
          | _ ->
            raise_s
              [%message
                "unexpected evaluation" (prim : Primitive.t) (tracers : _ Nonempty_list.t)]
        ;;
      end)
  in
  force interpreter
;;

let jvp ~f primals tangets =
  new_interpreter
    ~create_interpreter:jvp_interpreter
    ~global_data:()
    ~f:(fun interpreter ->
      let tracers_in =
        Nonempty_list.zip_exn primals tangets
        |> Nonempty_list.map ~f:(fun (primal, tangent) ->
          Value.of_tracer (T { tracer = { Jvp_tracer.primal; tangent }; interpreter }))
      in
      f tracers_in |> Nonempty_list.map ~f:(Value.full_raise interpreter))
;;

let jvp1 ~f primal tangent =
  match
    jvp
      ~f:(function
        | Nonempty_list.[ x ] -> [ f x ]
        | _ -> assert false)
      [ primal ]
      [ tangent ]
  with
  | [ { primal; tangent } ] -> primal, tangent
  | _ -> assert false
;;

let%expect_test "jvp" =
  let x = Value.of_float 3. in
  let dx = Value.of_float 1. in
  jvp1 ~f:sin x dx |> [%sexp_of: Value.Hide_id.t * Value.Hide_id.t] |> print_s;
  [%expect {| ((Tensor 0.14112000805986721) (Tensor -0.98999249660044542)) |}];
  cos x |> [%sexp_of: Value.Hide_id.t] |> print_s;
  [%expect {| (Tensor -0.98999249660044542) |}];
  let f x =
    let y = sin x * Value.of_float 2. in
    let z = -y + x in
    z
  in
  let x, dx = Value.of_float 3., Value.of_float 1. in
  let y, dy = jvp1 ~f x dx in
  [%sexp_of: Value.Hide_id.t * Value.Hide_id.t] (y, dy) |> print_s;
  [%expect {| ((Tensor 2.7177599838802657) (Tensor 2.9799849932008908)) |}];
  let deriv ~f x = jvp1 ~f x (Value.of_float 1.) |> snd in
  deriv ~f:sin x |> [%sexp_of: Value.Hide_id.t] |> print_s;
  [%expect {| (Tensor -0.98999249660044542) |}];
  deriv ~f:(deriv ~f:sin) x |> [%sexp_of: Value.Hide_id.t] |> print_s;
  [%expect {| (Tensor -0.14112000805986721) |}];
  deriv ~f:(deriv ~f:(deriv ~f:sin)) x |> [%sexp_of: Value.Hide_id.t] |> print_s;
  [%expect {| (Tensor 0.98999249660044542) |}];
  deriv ~f:(deriv ~f:(deriv ~f:(deriv ~f:sin))) x
  |> [%sexp_of: Value.Hide_id.t]
  |> print_s;
  [%expect {| (Tensor 0.14112000805986721) |}]
;;

let jaxpr_typecheck = Set_once.create ()

let abstract_eval (prim : Primitive.t) (inputs : Shaped_array.t Nonempty_list.t) =
  let unary_op { Shaped_array.dims } = { Shaped_array.dims } in
  let binary_op { Shaped_array.dims = dims1 } { Shaped_array.dims = dims2 } =
    if [%compare.equal: int array] dims1 dims2
    then { Shaped_array.dims = dims1 }
    else raise_s [%message "Mismatched dims" (dims1 : int array) (dims2 : int array)]
  in
  match prim, inputs with
  | Sin, [ x ] -> Nonempty_list.[ unary_op x ]
  | Cos, [ x ] -> [ unary_op x ]
  | Neg, [ x ] -> [ unary_op x ]
  | Add, [ x1; x2 ] -> [ binary_op x1 x2 ]
  | Sub, [ x1; x2 ] -> [ binary_op x1 x2 ]
  | Mul, [ x1; x2 ] -> [ binary_op x1 x2 ]
  | Xla_call { jaxpr; num_consts = _ }, args ->
    let jaxpr_type = Set_once.get_exn jaxpr_typecheck [%here] jaxpr in
    [%test_result: Shaped_array.t Nonempty_list.t]
      args
      ~expect:jaxpr_type.Jaxpr0.Type.in_types;
    jaxpr_type.out_types
  | _ -> raise_s [%message "Unexpected primitive" (prim : Primitive.t)]
;;

(* 'a list -> 'a Nonempty_list.t -> 'a Nonempty_list.t *)
let ( @* ) list nonempty_list =
  match list with
  | [] -> nonempty_list
  | x :: xs -> Nonempty_list.(x :: List.append xs (to_list nonempty_list))
;;

module Jaxpr = struct
  module Eqn = struct
    type t = Jaxpr0.Eqn.t =
      { prim : Primitive.t
      ; inputs : Atom.t Nonempty_list.t
      ; out_binders : Var.t Nonempty_list.t
      }
    [@@deriving sexp_of, compare]
  end

  module Type = struct
    type t = Jaxpr0.Type.t =
      { in_types : Shaped_array.t Nonempty_list.t
      ; out_types : Shaped_array.t Nonempty_list.t
      }
    [@@deriving sexp_of]
  end

  type t = Jaxpr0.t =
    { in_binders : Var.t Nonempty_list.t
    ; eqns : Eqn.t list
    ; outs : Atom.t Nonempty_list.t
    }
  [@@deriving sexp_of, compare]

  let typecheck =
    let typecheck_atom atom ~env =
      match atom with
      | Atom.Var var ->
        if not (Set.mem env (Var.id var))
        then
          raise_s
            [%message "Variable not found in env" (env : Var.Id.Set.t) (var : Var.t)];
        Var.shaped_array var
      | Lit tensor -> Shaped_array.of_tensor tensor
    in
    fun t ->
      let { in_binders; eqns; outs } = t in
      let env =
        Nonempty_list.fold in_binders ~init:Var.Id.Set.empty ~f:(fun env var ->
          match Set.mem env (Var.id var) with
          | true ->
            raise_s
              [%message
                "Duplicate variables found in in_binders"
                  (env : Var.Id.Set.t)
                  (var : Var.t)]
          | false -> Set.add env (Var.id var))
      in
      let env =
        List.fold eqns ~init:env ~f:(fun env eqn ->
          let { Eqn.prim; inputs; out_binders } = eqn in
          let input_types = Nonempty_list.map inputs ~f:(typecheck_atom ~env) in
          let out_types = abstract_eval prim input_types in
          Nonempty_list.zip_exn out_binders out_types
          |> Nonempty_list.iter ~f:(fun (var, expected) ->
            let actual = Var.shaped_array var in
            if not ([%compare.equal: Shaped_array.t] actual expected)
            then
              raise_s
                [%message
                  "Mismatched output types"
                    (actual : Shaped_array.t)
                    (expected : Shaped_array.t)]);
          Nonempty_list.fold out_binders ~init:env ~f:(fun env var ->
            if Set.mem env (Var.id var)
            then
              raise_s
                [%message
                  "Duplicate variables found in out_binders"
                    (env : Var.Id.Set.t)
                    (var : Var.t)];
            Set.add env (Var.id var)))
      in
      { Type.in_types = Nonempty_list.map in_binders ~f:Var.shaped_array
      ; out_types = Nonempty_list.map outs ~f:(typecheck_atom ~env)
      }
  ;;

  let () = Set_once.set_exn jaxpr_typecheck [%here] typecheck

  let eval t args =
    let { in_binders; eqns; outs } = t in
    let env =
      Nonempty_list.zip_exn in_binders args
      |> Nonempty_list.fold ~init:Var.Id.Map.empty ~f:(fun env (var, arg) ->
        Map.add_exn env ~key:(Var.id var) ~data:arg)
    in
    let read_atom atom ~env =
      match atom with
      | Atom.Var var -> Map.find_exn env (Var.id var)
      | Lit tensor -> Value.of_tensor tensor
    in
    let env =
      List.fold ~init:env eqns ~f:(fun env eqn ->
        let { Eqn.prim; inputs; out_binders } = eqn in
        let args = Nonempty_list.map inputs ~f:(read_atom ~env) in
        let outs = bind prim args in
        Nonempty_list.zip_exn out_binders outs
        |> Nonempty_list.fold ~init:env ~f:(fun env (var, out) ->
          Map.add_exn env ~key:(Var.id var) ~data:out))
    in
    Nonempty_list.map outs ~f:(read_atom ~env)
  ;;

  module Tracer = struct
    module Id = Id.Make (struct
        let name = "Jax.Jaxpr.Tracer"
      end)

    type t =
      { shaped_array : Shaped_array.t
      ; id : Id.t
      }
    [@@deriving sexp, compare]

    let create shaped_array = { shaped_array; id = Id.create () }
  end

  module Builder = struct
    type t =
      { mutable eqns : Eqn.t list
      ; mutable tracer_to_var : Var.t Tracer.Id.Map.t
      ; mutable const_tracers : Tracer.t Value.Id.Map.t
      ; mutable const_vals : Value.t Var.Id.Map.t
      ; mutable tracers : Tracer.t list
      }

    let create () =
      { eqns = []
      ; tracer_to_var = Tracer.Id.Map.empty
      ; const_tracers = Value.Id.Map.empty
      ; const_vals = Var.Id.Map.empty
      ; tracers = []
      }
    ;;

    let new_tracer t shaped_array =
      let tracer = Tracer.create shaped_array in
      t.tracers <- t.tracers @ [ tracer ];
      tracer
    ;;

    let add_eqn t eqn = t.eqns <- t.eqns @ [ eqn ]

    let add_var t (tracer : Tracer.t) =
      let var = Var.create tracer.shaped_array in
      t.tracer_to_var <- Map.add_exn t.tracer_to_var ~key:tracer.id ~data:var;
      var
    ;;

    let get_var t (tracer : Tracer.t) = Map.find_exn t.tracer_to_var tracer.id

    let add_const t (tracer : Tracer.t) value =
      let var = add_var t tracer in
      t.const_tracers <- Map.set t.const_tracers ~key:(Value.id value) ~data:tracer;
      t.const_vals <- Map.set t.const_vals ~key:(Var.id var) ~data:value;
      var
    ;;

    let build t in_tracers out_tracers =
      let { eqns; tracer_to_var; const_tracers = _; const_vals; tracers = _ } = t in
      let const_vars = Map.keys const_vals in
      let lookup_var (tracer : Tracer.t) = Map.find_exn tracer_to_var tracer.id in
      let in_binders = Nonempty_list.map in_tracers ~f:lookup_var in
      let t =
        { in_binders = List.map const_vars ~f:Var.lookup @* in_binders
        ; eqns
        ; outs =
            Nonempty_list.map out_tracers ~f:(fun tracer -> Atom.Var (lookup_var tracer))
        }
      in
      ignore (typecheck t : Type.t);
      (* Inline literals *)
      let const_vals, literals =
        Map.partition_map const_vals ~f:(fun value ->
          match Value.get value with
          | `Tracer _ -> First value
          | `Tensor tensor -> Second (Atom.Lit tensor))
      in
      let inline_literals atom =
        match atom with
        | Atom.Var var -> Map.find literals (Var.id var) |> Option.value ~default:atom
        | Lit _ -> atom
      in
      let const_vars, const_vals = Map.to_alist const_vals |> List.unzip in
      let t =
        { in_binders = List.map const_vars ~f:Var.lookup @* in_binders
        ; eqns =
            List.map eqns ~f:(fun eqn ->
              { eqn with inputs = Nonempty_list.map eqn.inputs ~f:inline_literals })
        ; outs = Nonempty_list.map t.outs ~f:inline_literals
        }
      in
      ignore (typecheck t : Type.t);
      t, const_vals
    ;;
  end

  let to_string t =
    let { in_binders; eqns; outs = _ } = t in
    let var_to_string var =
      let name = Var.id var |> Var.Id.to_string in
      let type_ =
        (Var.shaped_array var).dims
        |> Array.to_list
        |> List.map ~f:Int.to_string
        |> String.concat ~sep:","
      in
      [%string "%{name}:[%{type_}]"]
    in
    let in_binders =
      Nonempty_list.map in_binders ~f:var_to_string
      |> Nonempty_list.to_list
      |> String.concat ~sep:", "
    in
    let atom_to_string = function
      | Atom.Var var -> var_to_string var
      | Lit tensor -> [%sexp_of: Tensor.t] tensor |> Sexp.to_string
    in
    let eqns =
      List.map eqns ~f:(fun { Eqn.prim; inputs; out_binders } ->
        let lhs =
          Nonempty_list.map out_binders ~f:var_to_string
          |> Nonempty_list.to_list
          |> String.concat ~sep:" "
        in
        let rhs =
          let args =
            Nonempty_list.map inputs ~f:atom_to_string
            |> Nonempty_list.to_list
            |> String.concat ~sep:" "
          in
          [%string "%{prim#Primitive} %{args}"]
        in
        [%string "%{lhs} = %{rhs}"])
      |> String.concat ~sep:";\n  "
    in
    let outs =
      Nonempty_list.map t.outs ~f:atom_to_string
      |> Nonempty_list.to_list
      |> String.concat ~sep:", "
    in
    [%string "lambda %{in_binders} .\nlet %{eqns}\nin %{outs}"]
  ;;

  let () = Set_once.set_exn jaxpr_to_string [%here] to_string
end

let jaxpr_interpreter ~level (builder : Jaxpr.Builder.t)
  : (Jaxpr.Tracer.t, Jaxpr.Builder.t) Interpreter.t
  =
  let rec interpreter : (Jaxpr.Tracer.t, Jaxpr.Builder.t) Interpreter.t lazy_t =
    lazy
      (module struct
        type packed_tracer = Packed_tracer.t

        let name = "jaxpr_interpreter"

        module Tracer = struct
          type t = Jaxpr.Tracer.t [@@deriving sexp_of]
          type value = Value.t

          let aval (t : t) = Abstract_value.Shaped_array t.shaped_array

          let full_lower t =
            Value.of_tracer (T { tracer = t; interpreter = force interpreter })
          ;;
        end

        module Level = struct
          type global_data = Jaxpr.Builder.t

          let level = level
          let global_data = builder
        end

        let tracer_witness =
          Type_equal.Id.create ~name:"jaxpr_tracer" [%sexp_of: Jaxpr.Tracer.t]
        ;;

        let get_or_make_const_tracer (value : Value.t) =
          match Map.find builder.const_tracers (Value.id value) with
          | Some tracer -> tracer
          | None ->
            let dims = Value.get_aval value |> Abstract_value.dims in
            let tracer = Jaxpr.Builder.new_tracer builder { Shaped_array.dims } in
            let _ : Var.t = Jaxpr.Builder.add_const builder tracer value in
            tracer
        ;;

        let pure tensor = get_or_make_const_tracer (Value.of_tensor tensor)
        let lift packed_tracer = get_or_make_const_tracer (Value.of_tracer packed_tracer)

        let process_primitive prim (tracers : Jaxpr.Tracer.t Nonempty_list.t) =
          let shaped_arrays_in = Nonempty_list.map tracers ~f:(fun t -> t.shaped_array) in
          let shaped_arrays_out = abstract_eval prim shaped_arrays_in in
          let out_tracers =
            Nonempty_list.map shaped_arrays_out ~f:(fun shaped_array ->
              Jaxpr.Tracer.create shaped_array)
          in
          let inputs =
            Nonempty_list.map tracers ~f:(fun tracer ->
              Atom.Var (Jaxpr.Builder.get_var builder tracer))
          in
          let out_vars =
            Nonempty_list.map out_tracers ~f:(Jaxpr.Builder.add_var builder)
          in
          Jaxpr.Builder.add_eqn builder { Jaxpr.Eqn.prim; inputs; out_binders = out_vars };
          out_tracers
        ;;
      end)
  in
  force interpreter
;;

let new_arg
  ((module Interpreter) : (Jaxpr.Tracer.t, Jaxpr.Builder.t) Interpreter.t)
  shaped_array
  =
  let tracer = Jaxpr.Tracer.create shaped_array in
  let builder = Interpreter.Level.global_data in
  builder.tracer_to_var
  <- Map.add_exn builder.tracer_to_var ~key:tracer.id ~data:(Var.create shaped_array);
  tracer
;;

let make_jaxpr ~f args =
  let builder = Jaxpr.Builder.create () in
  new_interpreter
    ~create_interpreter:jaxpr_interpreter
    ~global_data:builder
    ~f:(fun interpreter ->
      new_dynamic_interpreter (Packed_interpreter.T interpreter) ~f:(fun () ->
        let tracers_in = Nonempty_list.map args ~f:(new_arg interpreter) in
        let outs =
          Nonempty_list.map tracers_in ~f:(fun tracer ->
            Value.of_tracer (Packed_tracer.T { tracer; interpreter }))
          |> f
        in
        let tracers_out = Nonempty_list.map outs ~f:(Value.full_raise interpreter) in
        let jaxpr, consts = Jaxpr.Builder.build builder tracers_in tracers_out in
        jaxpr, consts))
;;

let make_jaxpr1 ~f arg =
  make_jaxpr
    ~f:(function
      | Nonempty_list.[ x ] -> [ f x ]
      | _ -> assert false)
    [ arg ]
;;

let%expect_test "make_jaxpr" =
  let jaxpr, values =
    make_jaxpr1
      ~f:(fun x -> Value.of_float 2. * x)
      (Value.of_float 3. |> Value.get_aval |> Abstract_value.shaped_array)
  in
  print_s [%message "" (jaxpr : Jaxpr.t) (values : Value.t list)];
  [%expect
    {|
    ((jaxpr
      ((in_binders (((shaped_array ((dims ()))) (id (Jax.Var 0)))))
       (eqns
        (((prim Mul)
          (inputs ((Lit 2) (Var ((shaped_array ((dims ()))) (id (Jax.Var 0))))))
          (out_binders (((shaped_array ((dims ()))) (id (Jax.Var 2))))))))
       (outs ((Var ((shaped_array ((dims ()))) (id (Jax.Var 2))))))))
     (values ()))
    |}];
  Jaxpr.to_string jaxpr |> print_endline;
  [%expect
    {|
    lambda var_0:[] .
    let var_2:[] = mul 2 var_0:[]
    in var_2:[]
    |}];
  Jaxpr.typecheck jaxpr |> [%sexp_of: Jaxpr.Type.t] |> print_s;
  [%expect {| ((in_types (((dims ())))) (out_types (((dims ()))))) |}];
  make_jaxpr1
    ~f:(fun _ -> Value.of_float 2. * Value.of_float 2.)
    (Value.of_float 3. |> Value.get_aval |> Abstract_value.shaped_array)
  |> fst
  |> Jaxpr.to_string
  |> print_endline;
  [%expect {|
    lambda var_3:[] .
    let var_6:[] = mul 2 2
    in var_6:[]
    |}]
;;

let xla_subcomp jaxpr args ~builder =
  let { Jaxpr.in_binders; eqns; outs } = jaxpr in
  let env =
    Nonempty_list.zip_exn in_binders args
    |> Nonempty_list.fold ~init:Var.Id.Map.empty ~f:(fun env (var, xla_op) ->
      Map.add_exn env ~key:(Var.id var) ~data:xla_op)
  in
  let read_atom atom ~env =
    match atom with
    | Atom.Var var -> Map.find_exn env (Var.id var)
    | Lit tensor -> Tensor.to_xla_literal tensor |> Xla.Op.constant ~builder
  in
  let env =
    List.fold eqns ~init:env ~f:(fun env { prim; inputs; out_binders } ->
      let in_vals = Nonempty_list.map inputs ~f:(fun atom -> read_atom atom ~env) in
      let (out_vals : _ Nonempty_list.t) =
        match prim, in_vals with
        | Sin, [ x ] -> [ Xla.Op.sin x ]
        | Cos, [ x ] -> [ Xla.Op.cos x ]
        | Neg, [ x ] -> [ Xla.Op.neg x ]
        | Add, [ x1; x2 ] -> [ Xla.Op.add x1 x2 ]
        | Sub, [ x1; x2 ] -> [ Xla.Op.sub x1 x2 ]
        | Mul, [ x1; x2 ] -> [ Xla.Op.mul x1 x2 ]
        | _ -> raise_s [%message "unexpected xla compilation" (prim : Primitive.t)]
      in
      Nonempty_list.zip_exn out_binders out_vals
      |> Nonempty_list.fold ~init:env ~f:(fun env (var, xla_op) ->
        Map.add_exn env ~key:(Var.id var) ~data:xla_op))
  in
  Nonempty_list.map outs ~f:(fun atom -> read_atom atom ~env)
;;

let xla_callable jaxpr consts =
  ignore (Jaxpr.typecheck jaxpr : Jaxpr.Type.t);
  let in_avals =
    List.drop (Nonempty_list.to_list jaxpr.in_binders) (List.length consts)
  in
  let xla_builder = Xla.Builder.create ~name:"xla_call" in
  let xla_consts =
    List.map consts ~f:(fun tensor ->
      Tensor.to_xla_literal tensor |> Xla.Op.constant ~builder:xla_builder)
  in
  let xla_params =
    List.mapi in_avals ~f:(fun i var ->
      let var_id = Var.id var in
      let shaped_array = Var.shaped_array var in
      Xla.Op.parameter
        (Var.Id.to_string var_id)
        ~id:i
        ~ty:F64
        ~dims:shaped_array.dims
        ~builder:xla_builder)
  in
  let out =
    xla_subcomp
      jaxpr
      (Nonempty_list.of_list_exn (xla_consts @ xla_params))
      ~builder:xla_builder
    |> Nonempty_list.to_list
    |> Xla.Op.tuple ~builder:xla_builder
  in
  let xla_client = Xla.Client.cpu () in
  let xla_device = Xla.Client.addressable_devices xla_client |> List.hd_exn in
  let xla_exe = Xla.Computation.build ~root:out |> Xla.Executable.compile xla_client in
  Staged.stage (fun inputs ->
    let inputs =
      Nonempty_list.map inputs ~f:(fun tensor ->
        Tensor.to_xla_literal tensor |> Xla.Buffer.of_host_literal ~device:xla_device)
      |> Nonempty_list.to_array
    in
    let buffers = Xla.Executable.execute_b xla_exe inputs in
    let buffer = buffers.(0).(0) in
    Xla.Buffer.to_literal_sync buffer
    |> Xla.Literal.decompose_tuple
    |> Array.to_list
    |> Nonempty_list.of_list_exn
    |> Nonempty_list.map ~f:Tensor.of_xla_literal)
;;

let xla_callable =
  Memo.of_comparable
    (module struct
      module T = struct
        type t = Jaxpr.t * Tensor.t list [@@deriving compare, sexp_of]
      end

      include T
      include Comparable.Make_plain (T)
    end)
    (fun (jaxpr, consts) -> xla_callable jaxpr consts)
  |> Tuple2.curry
;;

(* TODO: Oof, such an ugly hack! *)
let () = Set_once.set_exn xla_callable' [%here] xla_callable

let jit ~f =
  Staged.stage (fun args ->
    let avals_in =
      Nonempty_list.map args ~f:(fun value ->
        Value.get_aval value |> Abstract_value.shaped_array)
    in
    let jaxpr, consts = make_jaxpr ~f avals_in in
    bind (Xla_call { jaxpr; num_consts = List.length consts }) args)
;;

let jit1 ~f =
  let f =
    jit ~f:(function
      | Nonempty_list.[ x ] -> [ f x ]
      | _ -> assert false)
    |> Staged.unstage
  in
  Staged.stage (fun x ->
    match f [ x ] with
    | [ z ] -> z
    | _ -> assert false)
;;

let jit2 ~f =
  let f =
    jit ~f:(function
      | Nonempty_list.[ x; y ] -> [ f x y ]
      | _ -> assert false)
    |> Staged.unstage
  in
  Staged.stage (fun x y ->
    match f [ x; y ] with
    | [ z ] -> z
    | _ -> assert false)
;;

let%expect_test "jit" =
  Core_unix.putenv ~key:"TF_CPP_MIN_LOG_LEVEL" ~data:"2";
  let f x y =
    print_endline "tracing!";
    sin x * cos y
  in
  let f_jitted = jit2 ~f |> Staged.unstage in
  f_jitted (Value.of_float 3.) (Value.of_float 4.)
  |> [%sexp_of: Value.Hide_id.t]
  |> print_s;
  [%expect {|
    tracing!
    (Tensor -0.092242193044553708)
    |}];
  (* TODO: cache isn't hitting, probably because of the various [Id.t]s
     sprinkled in various places :( *)
  let f_jitted = jit2 ~f |> Staged.unstage in
  f_jitted (Value.of_float 4.) (Value.of_float 5.)
  |> [%sexp_of: Value.Hide_id.t]
  |> print_s;
  [%expect {|
    tracing!
    (Tensor -0.21467624978306998)
    |}];
  let f x =
    let y = sin x * Value.of_float 2. in
    let z = -y + x in
    z
  in
  let deriv ~f x = jvp1 ~f x (Value.of_float 1.) |> snd in
  deriv ~f:(deriv ~f) (Value.of_float 3.) |> [%sexp_of: Value.Hide_id.t] |> print_s;
  Staged.unstage (jit1 ~f:(deriv ~f:(deriv ~f))) (Value.of_float 3.)
  |> [%sexp_of: Value.Hide_id.t]
  |> print_s;
  [%expect {|
    (Tensor 0.28224001611973443)
    (Tensor 0.28224001611973443)
    |}]
;;

let jvp_jaxpr (jaxpr : Jaxpr.t) =
  let backtrace = Backtrace.get () in
  let in_avals = Nonempty_list.map jaxpr.in_binders ~f:Var.shaped_array in
  make_jaxpr
    (Nonempty_list.append in_avals (Nonempty_list.to_list in_avals))
    ~f:(fun primals_and_tangents ->
      if !debug
      then
        print_s
          [%message
            "" (primals_and_tangents : Value.t Nonempty_list.t) (backtrace : Backtrace.t)];
      let primals, tangents =
        List.split_n
          (Nonempty_list.to_list primals_and_tangents)
          (Nonempty_list.length primals_and_tangents / 2)
        |> Tuple2.map ~f:Nonempty_list.of_list_exn
      in
      let primals, tangents =
        jvp primals tangents ~f:(fun primals -> Jaxpr.eval jaxpr primals)
        |> Nonempty_list.map ~f:(fun { primal; tangent } -> primal, tangent)
        |> Nonempty_list.unzip
      in
      Nonempty_list.append primals (Nonempty_list.to_list tangents))
;;

let () = Set_once.set_exn jvp_jaxpr' [%here] jvp_jaxpr

let%expect_test "jvp+jit" =
  let f x =
    let y = sin x * Value.of_float 2. in
    let z = -y + x in
    z
  in
  let x, dx = Value.of_float 3., Value.of_float 1. in
  let y, dy = jvp1 ~f:(Staged.unstage (jit1 ~f)) x dx in
  [%sexp_of: Value.Hide_id.t * Value.Hide_id.t] (y, dy) |> print_s;
  [%expect {| ((Tensor 2.7177599838802657) (Tensor 2.9799849932008908)) |}]
;;

module Partial_value = struct
  type t =
    | Known of
        { shaped_array : Shaped_array.t
        ; const : Value.t
        }
    | Unknown of { shaped_array : Shaped_array.t }
  [@@deriving sexp_of]
end

module rec Jaxpr_recipe_variant : sig
  type t =
    | Lambda_binding
    | Const_recipe of Value.t
    | Jaxpr_eqn of
        { prim : Primitive.t
        ; tracers_in : Partial_eval_tracer.t Nonempty_list.t
        ; avals_out : Shaped_array.t Nonempty_list.t
        ; tracer_refs_out : Partial_eval_tracer.t Nonempty_list.t lazy_t
        }
  [@@deriving sexp_of]
end = struct
  type t =
    | Lambda_binding
    | Const_recipe of Value.t
    | Jaxpr_eqn of
        { prim : Primitive.t
        ; tracers_in : Partial_eval_tracer.t Nonempty_list.t
        ; avals_out : Shaped_array.t Nonempty_list.t
        ; tracer_refs_out : Partial_eval_tracer.t Nonempty_list.t lazy_t
        }
  [@@deriving sexp_of]
end

and Jaxpr_recipe : sig
  module Id : Comparable.S

  type t =
    { id : Id.t
    ; variant : Jaxpr_recipe_variant.t
    }
  [@@deriving sexp_of]

  val create : Jaxpr_recipe_variant.t -> t
  val variant : t -> Jaxpr_recipe_variant.t
end = struct
  module Id = Id.Make (struct
      let name = "Jax.Jaxpr_recipe"
    end)

  type t =
    { id : Id.t
    ; variant : Jaxpr_recipe_variant.t
    }
  [@@deriving sexp_of, fields]

  let create variant = { id = Id.create (); variant }
end

and Partial_eval_tracer : sig
  module Id : Comparable.S

  type t =
    { partial_value : Partial_value.t
    ; recipe : Jaxpr_recipe.t option
    ; id : Id.t
    }
  [@@deriving sexp_of, compare, equal, hash]

  val shaped_array : t -> Shaped_array.t
  val create : partial_value:Partial_value.t -> recipe:Jaxpr_recipe.t option -> t
  val new_arg : Partial_value.t -> t
end = struct
  module Id = Id.Make (struct
      let name = "Jax.Partial_eval_tracer"
    end)

  type t =
    { partial_value : Partial_value.t
    ; recipe : Jaxpr_recipe.t option
    ; id : Id.t
    }
  [@@deriving sexp_of, fields]

  let compare = Comparable.lift [%compare: Id.t] ~f:id
  let equal = [%compare.equal: t]
  let hash t = Id.hash t.id
  let hash_fold_t state t = Id.hash_fold_t state t.id

  let shaped_array t =
    match t.partial_value with
    | Known { shaped_array; const = _ } -> shaped_array
    | Unknown { shaped_array } -> shaped_array
  ;;

  let create ~partial_value ~recipe = { partial_value; recipe; id = Id.create () }

  let new_arg partial_value =
    create ~partial_value ~recipe:(Some (Jaxpr_recipe.create Lambda_binding))
  ;;
end

let partial_eval_jaxpr' = Set_once.create ()

let merge_lists_tf =
  let rec go accum which xs ys ~sexp_of_t =
    match which, xs, ys with
    | [], [], [] -> List.rev accum
    | true :: which, x :: xs, _ -> go (x :: accum) which xs ys ~sexp_of_t
    | false :: which, _, y :: ys -> go (y :: accum) which xs ys ~sexp_of_t
    | _, _, _ ->
      raise_s [%message "unexpected extra elements" (xs : t list) (ys : t list)]
  in
  fun which xs ys ~sexp_of_t -> go [] which xs ys ~sexp_of_t
;;

let partial_eval_interpreter ~level () : (Partial_eval_tracer.t, unit) Interpreter.t =
  let rec interpreter : (Partial_eval_tracer.t, unit) Interpreter.t lazy_t =
    lazy
      (module struct
        type packed_tracer = Packed_tracer.t

        let name = "partial_eval_interpreter"

        module Tracer = struct
          type t = Partial_eval_tracer.t [@@deriving sexp_of]
          type value = Value.t

          let aval t = Abstract_value.Shaped_array (Partial_eval_tracer.shaped_array t)

          let full_lower (t : t) =
            match t.partial_value with
            | Known { shaped_array = _; const } -> Value.full_lower const
            | Unknown { shaped_array = _ } ->
              Value.of_tracer (T { tracer = t; interpreter = force interpreter })
          ;;
        end

        module Level = struct
          type global_data = unit

          let level = level
          let global_data = ()
        end

        let tracer_witness =
          Type_equal.Id.create
            ~name:"partial_eval_tracer"
            [%sexp_of: Partial_eval_tracer.t]
        ;;

        let tracer_of_value value =
          Partial_eval_tracer.create
            ~partial_value:
              (Known
                 { shaped_array = Value.get_aval value |> Abstract_value.shaped_array
                 ; const = value
                 })
            ~recipe:None
        ;;

        let pure tensor = tracer_of_value (Value.of_tensor tensor)
        let lift packed_tracer = tracer_of_value (Value.of_tracer packed_tracer)

        let instantiate_const (tracer : Partial_eval_tracer.t) =
          match tracer.partial_value with
          | Unknown { shaped_array = _ } -> tracer
          | Known { shaped_array; const } ->
            Partial_eval_tracer.create
              ~partial_value:(Unknown { shaped_array })
              ~recipe:(Some (Jaxpr_recipe.create (Const_recipe const)))
        ;;

        let process_primitive prim (tracers : Partial_eval_tracer.t Nonempty_list.t) =
          let are_all_tracers_known =
            Nonempty_list.for_all tracers ~f:(fun { partial_value; recipe = _; id = _ } ->
              match partial_value with
              | Known _ -> true
              | Unknown _ -> false)
          in
          if are_all_tracers_known
          then
            Nonempty_list.map tracers ~f:Tracer.full_lower
            |> bind prim
            |> Nonempty_list.map ~f:tracer_of_value
          else (
            match prim with
            | Xla_call { jaxpr; num_consts = _ } ->
              let in_unknowns =
                Nonempty_list.map tracers ~f:(fun { partial_value; recipe = _; id = _ } ->
                  match partial_value with
                  | Known _ -> false
                  | Unknown _ -> true)
              in
              let jaxpr1, jaxpr2, out_unknowns, num_res =
                Set_once.get_exn partial_eval_jaxpr' [%here] jaxpr ~in_unknowns
              in
              let known_values, unknown_tracers =
                Nonempty_list.to_list tracers
                |> List.partition_map
                     ~f:(fun ({ partial_value; recipe = _; id = _ } as tracer) ->
                       match partial_value with
                       | Known { const; shaped_array = _ } -> Either.First const
                       | Unknown _ -> Second tracer)
              in
              let outs1_res =
                bind
                  (Xla_call { jaxpr = jaxpr1; num_consts = 0 })
                  (Nonempty_list.of_list_exn known_values)
              in
              let outs1, res =
                List.split_n
                  (Nonempty_list.to_list outs1_res)
                  (Int.( - ) (Nonempty_list.length outs1_res) num_res)
              in
              let res_tracers =
                List.map res ~f:(fun value ->
                  Value.full_raise (force interpreter) value |> instantiate_const)
              in
              let rec outs2 =
                lazy
                  (Nonempty_list.map jaxpr2.Jaxpr.outs ~f:(fun atom ->
                     Partial_eval_tracer.create
                       ~partial_value:
                         (Partial_value.Unknown { shaped_array = Atom.shaped_array atom })
                       ~recipe:(Some (force eqn))))
              and eqn =
                lazy
                  (Jaxpr_recipe.create
                     (Jaxpr_recipe_variant.Jaxpr_eqn
                        { prim = Xla_call { jaxpr = jaxpr2; num_consts = 0 }
                        ; tracers_in =
                            Nonempty_list.of_list_exn (res_tracers @ unknown_tracers)
                        ; avals_out = Nonempty_list.map jaxpr2.outs ~f:Atom.shaped_array
                        ; tracer_refs_out = outs2
                        }))
              in
              merge_lists_tf
                (Nonempty_list.to_list out_unknowns)
                (force outs2 |> Nonempty_list.to_list)
                (List.map outs1 ~f:tracer_of_value)
                ~sexp_of_t:[%sexp_of: Partial_eval_tracer.t]
              |> Nonempty_list.of_list_exn
            | _ ->
              let tracers_in = Nonempty_list.map tracers ~f:instantiate_const in
              let avals_in =
                Nonempty_list.map tracers_in ~f:Partial_eval_tracer.shaped_array
              in
              let avals_out = abstract_eval prim avals_in in
              let rec tracers_out =
                lazy
                  (Nonempty_list.map avals_out ~f:(fun shaped_array ->
                     Partial_eval_tracer.create
                       ~partial_value:(Unknown { shaped_array })
                       ~recipe:(Some (force eqn))))
              and eqn =
                lazy
                  (Jaxpr_recipe.create
                     (Jaxpr_eqn
                        { prim; tracers_in; avals_out; tracer_refs_out = tracers_out }))
              in
              force tracers_out)
        ;;
      end)
  in
  force interpreter
;;

let topological_sort
  (type a)
  ((module Node : Topological_sort.Node with type t = a) as node_module)
  nodes
  ~parents
  =
  let nodes = List.stable_dedup nodes ~compare:Node.compare in
  let edges =
    List.concat_map nodes ~f:(fun child ->
      parents child
      |> List.map ~f:(fun parent -> { Topological_sort.Edge.from = parent; to_ = child }))
  in
  Topological_sort.sort node_module ~what:Nodes_and_edge_endpoints ~nodes ~edges
  |> Or_error.ok_exn
;;

let tracers_to_jaxpr
  ~(tracers_in : Partial_eval_tracer.t list)
  ~(tracers_out : Partial_eval_tracer.t list)
  =
  let tracer_to_var =
    List.map tracers_in ~f:(fun tracer ->
      tracer.id, Var.create (Partial_eval_tracer.shaped_array tracer))
    |> Partial_eval_tracer.Id.Map.of_alist_exn
  in
  let module Accum = struct
    type t =
      { tracer_to_var : Var.t Partial_eval_tracer.Id.Map.t
      ; constid_to_var : Var.t Value.Id.Map.t
      ; constvar_to_val : Value.t Var.Id.Map.t
      ; processed_eqns : Jaxpr_recipe.Id.Set.t
      ; eqns : Jaxpr.Eqn.t list
      }
  end
  in
  let { Accum.tracer_to_var
      ; constid_to_var = _
      ; constvar_to_val
      ; processed_eqns = _
      ; eqns
      }
    =
    if !debug then print_s [%message "" (tracers_in : Partial_eval_tracer.t list)];
    topological_sort
      (module Partial_eval_tracer)
      tracers_out
      ~parents:(fun { partial_value = _; recipe; id = _ } ->
        match Option.map recipe ~f:Jaxpr_recipe.variant with
        | Some (Jaxpr_eqn { prim = _; tracers_in; avals_out = _; tracer_refs_out = _ }) ->
          Nonempty_list.to_list tracers_in
        | Some (Lambda_binding | Const_recipe _) | None -> [])
    |> List.fold
         ~init:
           { Accum.tracer_to_var
           ; constid_to_var = Value.Id.Map.empty
           ; constvar_to_val = Var.Id.Map.empty
           ; processed_eqns = Jaxpr_recipe.Id.Set.empty
           ; eqns = []
           }
         ~f:
           (fun
             ({ tracer_to_var; constid_to_var; constvar_to_val; processed_eqns; eqns } as
              accum)
             { partial_value = _; recipe; id }
           ->
           let { Jaxpr_recipe.id = recipe_id; variant } = Option.value_exn recipe in
           match variant with
           | Lambda_binding ->
             assert (
               List.exists tracers_in ~f:(fun tracer ->
                 Partial_eval_tracer.Id.equal tracer.id id));
             accum
           | Const_recipe value ->
             let var = Map.find constid_to_var (Value.id value) in
             (match var with
              | Some var ->
                { accum with tracer_to_var = Map.add_exn tracer_to_var ~key:id ~data:var }
              | None ->
                let var =
                  Value.get_aval value |> Abstract_value.shaped_array |> Var.create
                in
                { accum with
                  tracer_to_var = Map.add_exn tracer_to_var ~key:id ~data:var
                ; constid_to_var =
                    Map.add_exn constid_to_var ~key:(Value.id value) ~data:var
                ; constvar_to_val =
                    Map.add_exn constvar_to_val ~key:(Var.id var) ~data:value
                })
           | Jaxpr_eqn
               { prim; tracers_in; avals_out; tracer_refs_out = (lazy tracer_refs_out) }
             ->
             (match Set.mem processed_eqns recipe_id with
              | true -> accum
              | false ->
                let tracer_to_var, eqn =
                  if !debug
                  then
                    print_s
                      [%message "" (tracer_to_var : Var.t Partial_eval_tracer.Id.Map.t)];
                  let inputs =
                    Nonempty_list.map tracers_in ~f:(fun tracer ->
                      Map.find_exn tracer_to_var tracer.id |> Atom.Var)
                  in
                  let out_binders = Nonempty_list.map avals_out ~f:Var.create in
                  let tracer_to_var =
                    Nonempty_list.zip_exn tracer_refs_out out_binders
                    |> Nonempty_list.fold
                         ~init:tracer_to_var
                         ~f:(fun tracer_to_var (tracer, var) ->
                           Map.add_exn tracer_to_var ~key:tracer.id ~data:var)
                  in
                  tracer_to_var, { Jaxpr.Eqn.prim; inputs; out_binders }
                in
                { accum with
                  tracer_to_var
                ; processed_eqns = Set.add processed_eqns recipe_id
                ; eqns = eqn :: eqns
                }))
  in
  let constvars, constvals = Map.to_alist constvar_to_val |> List.unzip in
  let in_binders =
    List.map constvars ~f:Var.lookup
    @ List.map tracers_in ~f:(fun tracer -> Map.find_exn tracer_to_var tracer.id)
    |> Nonempty_list.of_list_exn
  in
  let out_vars =
    List.map tracers_out ~f:(fun tracer ->
      Map.find_exn tracer_to_var tracer.id |> Atom.Var)
    |> Nonempty_list.of_list_exn
  in
  let jaxpr = { Jaxpr.in_binders; eqns = List.rev eqns; outs = out_vars } in
  ignore (Jaxpr.typecheck jaxpr : Jaxpr.Type.t);
  jaxpr, constvals
;;

let partial_eval ~f pvals_in =
  new_interpreter
    ~create_interpreter:partial_eval_interpreter
    ~global_data:()
    ~f:(fun interpreter ->
      let tracers_in = Nonempty_list.map pvals_in ~f:Partial_eval_tracer.new_arg in
      let outs =
        Nonempty_list.map tracers_in ~f:(fun tracer ->
          Value.of_tracer (T { tracer; interpreter }))
        |> f
      in
      let tracers_out = Nonempty_list.map outs ~f:(Value.full_raise interpreter) in
      let partial_values =
        Nonempty_list.map tracers_out ~f:(fun tracer -> tracer.partial_value)
      in
      let filter_out_known =
        Nonempty_list.filter ~f:(fun (tracer : Partial_eval_tracer.t) ->
          match tracer.partial_value with
          | Known _ -> false
          | Unknown _ -> true)
      in
      let unknown_tracers_in = filter_out_known tracers_in in
      let unknown_tracers_out = filter_out_known tracers_out in
      let jaxpr, consts =
        tracers_to_jaxpr ~tracers_in:unknown_tracers_in ~tracers_out:unknown_tracers_out
      in
      jaxpr, partial_values, consts)
;;

let linearize ~f primals_in =
  let pvals_in =
    Nonempty_list.concat
      [ Nonempty_list.map primals_in ~f:(fun primal ->
          Partial_value.Known
            { shaped_array = Value.get_aval primal |> Abstract_value.shaped_array
            ; const = primal
            })
      ; Nonempty_list.map primals_in ~f:(fun primal ->
          Partial_value.Unknown
            { shaped_array = Value.get_aval primal |> Abstract_value.shaped_array })
      ]
  in
  let jaxpr, pvals_out, consts =
    partial_eval
      ~f:(fun primals_tangents_in ->
        let primals_in, tangents_in =
          List.split_n
            (Nonempty_list.to_list primals_tangents_in)
            (Nonempty_list.length primals_tangents_in / 2)
          |> Tuple2.map ~f:Nonempty_list.of_list_exn
        in
        let primals_out, tangents_out =
          jvp ~f primals_in tangents_in
          |> Nonempty_list.map ~f:(fun { primal; tangent } -> primal, tangent)
          |> Nonempty_list.unzip
        in
        Nonempty_list.concat [ primals_out; tangents_out ])
      pvals_in
  in
  let primal_pvals =
    List.take (Nonempty_list.to_list pvals_out) (Nonempty_list.length pvals_out / 2)
  in
  let primals_out =
    List.map primal_pvals ~f:(function
      | Known { shaped_array = _; const } -> const
      | Unknown { shaped_array = _ } ->
        raise_s
          [%message
            "unexpected unknown priaml partial value"
              (primal_pvals : Partial_value.t list)])
  in
  let f_lin tangents = Jaxpr.eval jaxpr (consts @* tangents) in
  primals_out, f_lin
;;

let linearize1 ~f value =
  match
    linearize
      ~f:(function
        | [ x ] -> [ f x ]
        | _ -> assert false)
      (Nonempty_list.singleton value)
  with
  | [ y ], f_lin ->
    ( y
    , fun t ->
        (match f_lin (Nonempty_list.singleton t) with
         | [ t ] -> t
         | _ -> assert false) )
  | _ -> assert false
;;

let%expect_test "linearize" =
  let y, sin_lin = linearize1 ~f:sin (Value.of_float 3.) in
  [%sexp_of: Value.Hide_id.t * Value.Hide_id.t] (y, sin (Value.of_float 3.)) |> print_s;
  [%expect {| ((Tensor 0.14112000805986721) (Tensor 0.14112000805986721)) |}];
  [%sexp_of: Value.Hide_id.t * Value.Hide_id.t]
    (sin_lin (Value.of_float 1.), cos (Value.of_float 3.))
  |> print_s;
  [%expect {| ((Tensor -0.98999249660044542) (Tensor -0.98999249660044542)) |}]
;;

let typecheck_partial_eval_jaxpr jaxpr ~unks_in ~unks_out ~jaxpr1 ~jaxpr2 =
  let jaxpr_type =
    (* (a1, a2) -> (b1, b2) *)
    Jaxpr.typecheck jaxpr
  in
  let jaxpr1_type =
    (* a1 -> (b1, res) *)
    Jaxpr.typecheck jaxpr1
  in
  let jaxpr2_type =
    (* (res, a2) -> b2 *)
    Jaxpr.typecheck jaxpr2
  in
  let a1, a2 =
    Nonempty_list.zip_exn jaxpr_type.in_types unks_in
    |> Nonempty_list.to_list
    |> List.partition_map ~f:(fun (type_, is_unknown) ->
      match is_unknown with
      | true -> Either.Second type_
      | false -> First type_)
  in
  let b1, b2 =
    Nonempty_list.zip_exn jaxpr_type.out_types unks_out
    |> Nonempty_list.to_list
    |> List.partition_map ~f:(fun (type_, is_unknown) ->
      match is_unknown with
      | true -> Either.Second type_
      | false -> First type_)
  in
  let b1_, res =
    List.split_n (Nonempty_list.to_list jaxpr1_type.out_types) (List.length b1)
  in
  let res_, a2_ =
    List.split_n (Nonempty_list.to_list jaxpr2_type.in_types) (List.length res)
  in
  let b2_ = Nonempty_list.to_list jaxpr2_type.out_types in
  [%test_result: Shaped_array.t list]
    (Nonempty_list.to_list jaxpr1_type.in_types)
    ~expect:a1;
  [%test_result: Shaped_array.t list] b1_ ~expect:b1;
  [%test_result: Shaped_array.t list] res_ ~expect:res;
  [%test_result: Shaped_array.t list] a2_ ~expect:a2;
  [%test_result: Shaped_array.t list] b2_ ~expect:b2
;;

let rec partial_eval_jaxpr jaxpr ~in_unknowns =
  let { Jaxpr.in_binders; eqns; outs } = jaxpr in
  let env =
    Nonempty_list.map2_exn in_binders in_unknowns ~f:(fun var unknown ->
      Var.id var, unknown)
    |> Nonempty_list.to_list
    |> Var.Id.Map.of_alist_exn
  in
  let module Accum = struct
    type t =
      { env : bool Var.Id.Map.t
      ; residuals : Var.Id.Set.t
      ; eqns1 : Jaxpr.Eqn.t list
      ; eqns2 : Jaxpr.Eqn.t list
      }
  end
  in
  let is_unknown ~env = function
    | Atom.Var var -> Map.find_exn env (Var.id var)
    | Lit _ -> false
  in
  let { Accum.env; residuals; eqns1; eqns2 } =
    List.fold
      eqns
      ~init:{ Accum.env; residuals = Var.Id.Set.empty; eqns1 = []; eqns2 = [] }
      ~f:
        (fun
          ({ env; residuals; eqns1; eqns2 } as accum)
          ({ prim; inputs; out_binders } as eqn)
        ->
        let unks_in = Nonempty_list.map inputs ~f:(is_unknown ~env) in
        match prim with
        | Xla_call { jaxpr; num_consts } ->
          assert (num_consts = 0);
          let eqn1, eqn2, unks_out, res =
            (* xla_call_peval_eqn *)
            let jaxpr1, jaxpr2, unks_out, num_res =
              partial_eval_jaxpr jaxpr ~in_unknowns:unks_in
            in
            let ins1, ins2 =
              Nonempty_list.zip_exn inputs unks_in
              |> Nonempty_list.to_list
              |> List.partition_map ~f:(fun (v, unk) ->
                match unk with
                | true -> Either.Second v
                | false -> First v)
              |> Tuple2.map ~f:Nonempty_list.of_list_exn
            in
            let out_binders1, out_binders2 =
              Nonempty_list.zip_exn out_binders unks_out
              |> Nonempty_list.to_list
              |> List.partition_map ~f:(fun (v, unk) ->
                match unk with
                | true -> Either.Second v
                | false -> First v)
              |> Tuple2.map ~f:Nonempty_list.of_list_exn
            in
            let residuals =
              List.take (Nonempty_list.to_list jaxpr2.Jaxpr.in_binders) num_res
            in
            let eqn1 =
              { Jaxpr.Eqn.prim = Xla_call { jaxpr = jaxpr1; num_consts = 0 }
              ; inputs = ins1
              ; out_binders = Nonempty_list.append out_binders1 residuals
              }
            in
            let eqn2 =
              { Jaxpr.Eqn.prim = Xla_call { jaxpr = jaxpr2; num_consts = 0 }
              ; inputs = List.map residuals ~f:(fun var -> Atom.Var var) @* ins2
              ; out_binders = out_binders2
              }
            in
            eqn1, eqn2, unks_out, List.map residuals ~f:Var.id |> Var.Id.Set.of_list
          in
          let eqns1 = eqn1 :: eqns1 in
          let eqns2 = eqn2 :: eqns2 in
          let residuals = Set.union residuals res in
          let env =
            Nonempty_list.zip_exn out_binders unks_out
            |> Nonempty_list.fold ~init:env ~f:(fun env (var, unk) ->
              Map.add_exn env ~key:(Var.id var) ~data:unk)
          in
          { eqns1; eqns2; residuals; env }
        | _ ->
          (match Nonempty_list.exists unks_in ~f:Fn.id with
           | true ->
             let residuals =
               Nonempty_list.zip_exn unks_in inputs
               |> Nonempty_list.fold ~init:residuals ~f:(fun residuals (unk, v) ->
                 match v with
                 | Atom.Var var when not unk -> Set.add residuals (Var.id var)
                 | Var _ | Lit _ -> residuals)
             in
             let eqns2 = { Jaxpr.Eqn.prim; inputs; out_binders } :: eqns2 in
             let env =
               Nonempty_list.fold out_binders ~init:env ~f:(fun env var ->
                 Map.add_exn env ~key:(Var.id var) ~data:true)
             in
             { accum with env; residuals; eqns2 }
           | false ->
             let eqns1 = eqn :: eqns1 in
             let env =
               Nonempty_list.fold out_binders ~init:env ~f:(fun env var ->
                 Map.add_exn env ~key:(Var.id var) ~data:false)
             in
             { accum with env; eqns1 }))
  in
  let out_unknowns = Nonempty_list.map outs ~f:(is_unknown ~env) in
  let ins1, ins2 =
    Nonempty_list.zip_exn in_binders in_unknowns
    |> Nonempty_list.to_list
    |> List.partition_map ~f:(fun (var, is_unknown) ->
      match is_unknown with
      | true -> Either.Second var
      | false -> First var)
    |> Tuple2.map ~f:Nonempty_list.of_list_exn
  in
  let outs1, outs2 =
    Nonempty_list.zip_exn outs out_unknowns
    |> Nonempty_list.to_list
    |> List.partition_map ~f:(fun (var, is_unknown) ->
      match is_unknown with
      | true -> Either.Second var
      | false -> First var)
  in
  let residuals = Set.to_list residuals |> List.map ~f:Var.lookup in
  let jaxpr1 =
    { Jaxpr.in_binders = ins1
    ; eqns = List.rev eqns1
    ; outs =
        Nonempty_list.of_list_exn (outs1 @ List.map residuals ~f:(fun var -> Atom.Var var))
    }
  in
  let jaxpr2 =
    { Jaxpr.in_binders = residuals @* ins2
    ; eqns = List.rev eqns2
    ; outs = Nonempty_list.of_list_exn outs2
    }
  in
  typecheck_partial_eval_jaxpr
    jaxpr
    ~unks_in:in_unknowns
    ~unks_out:out_unknowns
    ~jaxpr1
    ~jaxpr2;
  jaxpr1, jaxpr2, out_unknowns, List.length residuals
;;

let () = Set_once.set_exn partial_eval_jaxpr' [%here] partial_eval_jaxpr

let%expect_test "jit+linearize" =
  let f x =
    let y = sin x * Value.of_float 2. in
    let z = -y + x in
    z
  in
  let f_jitted = jit1 ~f |> Staged.unstage in
  let y, f_lin = linearize1 ~f:f_jitted (Value.of_float 3.) in
  let y_dot = f_lin (Value.of_float 1.) in
  [%sexp_of: Value.Hide_id.t * Value.Hide_id.t] (y, y_dot) |> print_s;
  [%expect {| ((Tensor 2.7177599838802657) (Tensor 2.9799849932008908)) |}]
;;
