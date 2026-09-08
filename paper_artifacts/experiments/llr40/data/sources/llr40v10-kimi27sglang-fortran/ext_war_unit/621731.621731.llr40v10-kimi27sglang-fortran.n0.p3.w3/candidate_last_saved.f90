subroutine ext_war_unit_fp64(a, b, n, workspace, workspace_size) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value, intent(in) :: a
  type(c_ptr), value, intent(in) :: b
  integer(c_int64_t), value, intent(in) :: n
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), pointer, contiguous :: pa(:)
  real(c_double), pointer, contiguous :: pb(:)
  integer(c_int64_t) :: i
  integer :: nt, t
  integer(c_int64_t) :: total, chunk, start_idx, end_idx
  integer, parameter :: MAXT = 64
  integer :: ready(MAXT)
  real(c_double) :: right, left
  integer :: signal
  call c_f_pointer(a, pa, [n])
  call c_f_pointer(b, pb, [n])
  total = n - 1
  if (total <= 4096_c_int64_t) then
    do i = 1, total
      pa(i) = pa(i + 1) + pb(i)
    end do
  else
    nt = omp_get_max_threads()
    if (nt > MAXT) nt = MAXT
    ready(1:nt) = 0
    !$omp parallel default(none) shared(pa, pb, total, nt, ready) private(t, chunk, start_idx, end_idx, i, right, left, signal)
      t = omp_get_thread_num()
      if (t < nt) then
        chunk = ((total + int(nt, c_int64_t) - 1) / int(nt, c_int64_t) + 7) / 8 * 8
        start_idx = int(t, c_int64_t) * chunk + 1
        end_idx = min(start_idx + chunk - 1, total)
        if (start_idx <= end_idx) then
          right = pa(end_idx + 1)
          !$omp atomic write seq_cst
          ready(t + 1) = 1
          !$omp end atomic
          if (start_idx < end_idx) then
            left = pa(start_idx + 1)
            !$omp simd simdlen(8) aligned(pa, pb : 64)
            do i = start_idx + 1, end_idx - 1
              pa(i) = pa(i + 1) + pb(i)
            end do
            !$omp end simd
            if (t > 0) then
              do
                !$omp atomic read seq_cst
                signal = ready(t)
                !$omp end atomic
                if (signal == 1) exit
              end do
            end if
            pa(start_idx) = left + pb(start_idx)
          else
            if (t > 0) then
              do
                !$omp atomic read seq_cst
                signal = ready(t)
                !$omp end atomic
                if (signal == 1) exit
              end do
            end if
          end if
          pa(end_idx) = right + pb(end_idx)
        end if
      end if
    !$omp end parallel
  end if
end subroutine ext_war_unit_fp64
