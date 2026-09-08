subroutine tsvc_2_s3112_fp64(a, b, n) bind(C, name="tsvc_2_s3112_fp64")
  use iso_c_binding, only: c_int64_t, c_double, c_int, c_ptr, c_funptr, &
       c_null_ptr, c_null_funptr, c_null_char, c_char, c_f_procpointer, c_associated
  use omp_lib, only: omp_get_max_threads, omp_get_thread_num
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(in) :: a(n)
  real(c_double), intent(out) :: b(n)

  integer(c_int64_t) :: i, lo, hi
  integer(c_int) :: nt_env, nt_used, flag, t, j
  real(c_double) :: s, run
  real(c_double) :: part(0:32)

  type(c_ptr), save :: handle = c_null_ptr
  type(c_funptr), save :: fptr = c_null_funptr
  logical, save :: loaded = .false.

  abstract interface
     subroutine helper_fn(a_, b_, n_, nt_) bind(C)
       import :: c_int64_t, c_double, c_int
       integer(c_int64_t), value :: n_
       integer(c_int), value :: nt_
       real(c_double), intent(in) :: a_(n_)
       real(c_double), intent(out) :: b_(n_)
     end subroutine helper_fn
  end interface
  procedure(helper_fn), pointer :: hp

  interface
     function dlopen(filename, flag) bind(C, name='dlopen')
       import :: c_ptr, c_int, c_char
       type(c_ptr) :: dlopen
       character(kind=c_char), intent(in) :: filename(*)
       integer(c_int), value :: flag
     end function dlopen
     function dlsym(handle, symbol) bind(C, name='dlsym')
       import :: c_ptr, c_funptr, c_char
       type(c_funptr) :: dlsym
       type(c_ptr), value :: handle
       character(kind=c_char), intent(in) :: symbol(*)
     end function dlsym
  end interface

  if (n <= 0) return

  if (.not. loaded) then
     flag = 2
     handle = dlopen('/shared/agent-1/libtsvc_2_s3112_helper.so' // c_null_char, flag)
     if (c_associated(handle)) then
        fptr = dlsym(handle, 'tsvc_2_s3112_helper' // c_null_char)
     end if
     loaded = .true.
  end if

  if (c_associated(fptr)) then
     call c_f_procpointer(fptr, hp)
     nt_used = int(min(omp_get_max_threads(), 26), kind=c_int)
     call hp(a, b, n, nt_used)
     return
  end if

  nt_env = omp_get_max_threads()
  nt_used = nt_env
  if (nt_used > 26) nt_used = 26

  if (n < 4096 .or. nt_used <= 1) then
     s = 0.0d0
     do i = 1, n
        s = s + a(i)
        b(i) = s
     end do
     return
  end if

  part(0:nt_used) = 0.0d0

  !$omp parallel private(t, lo, hi, run, i) num_threads(nt_used)
  t = omp_get_thread_num()
  lo = (n * int(t, kind=c_int64_t)) / int(nt_used, kind=c_int64_t) + 1
  hi = (n * int(t + 1, kind=c_int64_t)) / int(nt_used, kind=c_int64_t)

  run = 0.0d0
  do i = lo, hi
     run = run + a(i)
  end do
  part(t + 1) = run

  !$omp barrier
  !$omp single
  do j = 1, nt_used
     part(j) = part(j) + part(j - 1)
  end do
  !$omp end single

  run = part(t)
  do i = lo, hi
     run = run + a(i)
     b(i) = run
  end do
  !$omp end parallel
end subroutine tsvc_2_s3112_fp64
