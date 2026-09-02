subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none

  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer, parameter :: MAXT = 128
  integer(c_int64_t) :: n, nt, t, lo, hi, i, k, best_idx
  real(c_double) :: best_val, val
  real(c_double), target :: local_val_fixed(MAXT)
  integer(c_int64_t), target :: local_idx_fixed(MAXT)
  real(c_double), allocatable, target :: local_val_dyn(:)
  integer(c_int64_t), allocatable, target :: local_idx_dyn(:)
  real(c_double), pointer :: local_val(:)
  integer(c_int64_t), pointer :: local_idx(:)

  n = LEN_1D
  if (n <= 0_c_int64_t) then
     out_value = 0.0_c_double
     out_index = 0_c_int64_t
     return
  end if

  nt = omp_get_max_threads()
  if (n < 1024_c_int64_t .or. nt <= 1) then
     best_val = a(1)
     best_idx = 1_c_int64_t
     do i = 2_c_int64_t, n
        if (a(i) > best_val) then
           best_val = a(i)
           best_idx = i
        end if
     end do
     out_value = best_val
     out_index = best_idx
     return
  end if

  if (nt <= MAXT) then
     local_val => local_val_fixed
     local_idx => local_idx_fixed
  else
     allocate(local_val_dyn(nt), local_idx_dyn(nt))
     local_val => local_val_dyn
     local_idx => local_idx_dyn
  end if

  !$omp parallel private(t, lo, hi, best_val, best_idx, i)
  t = omp_get_thread_num()
  lo = (n * t) / nt + 1_c_int64_t
  hi = (n * (t + 1_c_int64_t)) / nt
  best_val = a(lo)
  best_idx = lo
  do i = lo + 1_c_int64_t, hi
     if (a(i) > best_val) then
        best_val = a(i)
        best_idx = i
     end if
  end do
  local_val(t + 1_c_int64_t) = best_val
  local_idx(t + 1_c_int64_t) = best_idx
  !$omp end parallel

  best_val = local_val(1)
  best_idx = local_idx(1)
  do k = 2_c_int64_t, nt
     val = local_val(k)
     if (val > best_val) then
        best_val = val
        best_idx = local_idx(k)
     end if
  end do

  out_value = best_val
  out_index = best_idx
end subroutine argmax_with_index_fp64
